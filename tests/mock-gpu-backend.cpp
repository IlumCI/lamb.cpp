#include "mock-gpu-backend.h"

#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml-cpu.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

//
// device
//

struct mock_dev_ctx {
    int64_t copy_delay_us;
    int64_t offload_min_batch;

    ggml_backend_buffer_type buft;
    ggml_backend_reg         reg;

    std::atomic<int64_t> n_set_async  {0};
    std::atomic<int64_t> n_graphs     {0};
    std::atomic<int64_t> n_instances  {0};
    std::atomic<int64_t> n_overlap_us {0};
    std::atomic<int>     running_graphs {0};
};

static mock_dev_ctx * dev_ctx(ggml_backend_dev_t dev) {
    return (mock_dev_ctx *) dev->context;
}

//
// buffer: plain host memory that the scheduler must treat as device memory
//

struct mock_buffer_ctx {
    void * data;
};

static void mock_buffer_free(ggml_backend_buffer_t buffer) {
    mock_buffer_ctx * ctx = (mock_buffer_ctx *) buffer->context;
    ggml_aligned_free(ctx->data, buffer->size);
    delete ctx;
}

static void * mock_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((mock_buffer_ctx *) buffer->context)->data;
}

static void mock_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
}

static void mock_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
}

static void mock_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
}

static bool mock_buffer_cpy_tensor(ggml_backend_buffer_t, const ggml_tensor * src, ggml_tensor * dst) {
    if (src->buffer && ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void mock_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(((mock_buffer_ctx *) buffer->context)->data, value, buffer->size);
}

static const ggml_backend_buffer_i mock_buffer_iface = {
    /* .free_buffer   = */ mock_buffer_free,
    /* .get_base      = */ mock_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ mock_buffer_memset_tensor,
    /* .set_tensor    = */ mock_buffer_set_tensor,
    /* .get_tensor    = */ mock_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ mock_buffer_cpy_tensor,
    /* .clear         = */ mock_buffer_clear,
    /* .reset         = */ nullptr,
};

static const char * mock_buft_get_name(ggml_backend_buffer_type_t) {
    return "MOCK0";
}

static ggml_backend_buffer_t mock_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    size = std::max<size_t>(size, 64);
    void * data = ggml_aligned_malloc(size);
    if (data == nullptr) {
        return nullptr;
    }
    // poison new memory, so that reading a tensor before it was written gives NaN
    memset(data, 0xff, size);
    return ggml_backend_buffer_init(buft, mock_buffer_iface, new mock_buffer_ctx { data }, size);
}

static size_t mock_buft_get_alignment(ggml_backend_buffer_type_t) {
    return 64;
}

static bool mock_buft_is_host(ggml_backend_buffer_type_t) {
    return false;
}

static const ggml_backend_buffer_type_i mock_buft_iface = {
    /* .get_name         = */ mock_buft_get_name,
    /* .alloc_buffer     = */ mock_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ mock_buft_get_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ nullptr,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ mock_buft_is_host,
};

//
// events
//

struct mock_event {
    std::mutex              m;
    std::condition_variable cv;
    uint64_t                recorded = 0;
    uint64_t                done     = 0;
};

//
// backend instance: one stream, run by a worker thread
//

struct mock_stream {
    ggml_backend_dev_t dev;
    ggml_backend_t     cpu;

    std::thread                       worker;
    std::mutex                        m;
    std::condition_variable           cv;
    std::condition_variable           cv_idle;
    std::deque<std::function<void()>> queue;
    bool                              busy = false;
    bool                              stop = false;

    void push(std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> lock(m);
            queue.push_back(std::move(fn));
        }
        cv.notify_one();
    }

    void wait_idle() {
        std::unique_lock<std::mutex> lock(m);
        cv_idle.wait(lock, [&] { return queue.empty() && !busy; });
    }

    void run() {
        for (;;) {
            std::function<void()> fn;
            {
                std::unique_lock<std::mutex> lock(m);
                cv.wait(lock, [&] { return stop || !queue.empty(); });
                if (queue.empty()) {
                    return;
                }
                fn = std::move(queue.front());
                queue.pop_front();
                busy = true;
            }
            fn();
            {
                std::lock_guard<std::mutex> lock(m);
                busy = false;
            }
            cv_idle.notify_all();
        }
    }
};

static mock_stream * stream_of(ggml_backend_t backend) {
    return (mock_stream *) backend->context;
}

static const char * mock_backend_get_name(ggml_backend_t) {
    return "MOCK0";
}

static void mock_backend_free(ggml_backend_t backend) {
    mock_stream * s = stream_of(backend);
    s->wait_idle();
    {
        std::lock_guard<std::mutex> lock(s->m);
        s->stop = true;
    }
    s->cv.notify_all();
    s->worker.join();
    ggml_backend_free(s->cpu);
    delete s;
    delete backend;
}

static void mock_backend_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    mock_stream  * s = stream_of(backend);
    mock_dev_ctx * d = dev_ctx(s->dev);
    d->n_set_async++;
    char * dst = (char *) tensor->data + offset;
    s->push([d, dst, data, size] {
        if (d->copy_delay_us > 0) {
            const bool overlap_start = d->running_graphs.load() > 0;
            std::this_thread::sleep_for(std::chrono::microseconds(d->copy_delay_us));
            if (overlap_start || d->running_graphs.load() > 0) {
                d->n_overlap_us += d->copy_delay_us;
            }
        }
        memcpy(dst, data, size);
    });
}

static void mock_backend_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const char * src = (const char *) tensor->data + offset;
    stream_of(backend)->push([src, data, size] {
        memcpy(data, src, size);
    });
}

static void mock_backend_synchronize(ggml_backend_t backend) {
    stream_of(backend)->wait_idle();
}

static enum ggml_status mock_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    mock_stream  * s = stream_of(backend);
    mock_dev_ctx * d = dev_ctx(s->dev);
    d->n_graphs++;
    // the scheduler keeps the node arrays alive until the next sync, the struct may be on its stack
    ggml_cgraph g = *cgraph;
    ggml_backend_t cpu = s->cpu;
    s->push([d, g, cpu]() mutable {
        d->running_graphs++;
        ggml_backend_graph_compute(cpu, &g);
        d->running_graphs--;
    });
    return GGML_STATUS_SUCCESS;
}

static void mock_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    mock_event * ev = (mock_event *) event->context;
    uint64_t t;
    {
        std::lock_guard<std::mutex> lock(ev->m);
        t = ++ev->recorded;
    }
    stream_of(backend)->push([ev, t] {
        {
            std::lock_guard<std::mutex> lock(ev->m);
            ev->done = std::max(ev->done, t);
        }
        ev->cv.notify_all();
    });
}

static void mock_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    mock_event * ev = (mock_event *) event->context;
    uint64_t target;
    {
        std::lock_guard<std::mutex> lock(ev->m);
        target = ev->recorded;
    }
    stream_of(backend)->push([ev, target] {
        std::unique_lock<std::mutex> lock(ev->m);
        ev->cv.wait(lock, [&] { return ev->done >= target; });
    });
}

static const ggml_backend_i mock_backend_iface = {
    /* .get_name            = */ mock_backend_get_name,
    /* .free                = */ mock_backend_free,
    /* .set_tensor_async    = */ mock_backend_set_tensor_async,
    /* .get_tensor_async    = */ mock_backend_get_tensor_async,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ mock_backend_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ mock_backend_graph_compute,
    /* .event_record        = */ mock_backend_event_record,
    /* .event_wait          = */ mock_backend_event_wait,
    /* .graph_optimize      = */ nullptr,
};

static ggml_guid_t mock_guid() {
    static ggml_guid guid = { 0x6d, 0x6f, 0x63, 0x6b, 0x2d, 0x67, 0x70, 0x75, 0x2d, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64 };
    return &guid;
}

//
// device interface
//

static const char * mock_dev_get_name(ggml_backend_dev_t) {
    return "MOCK0";
}

static const char * mock_dev_get_description(ggml_backend_dev_t) {
    return "mock asynchronous GPU for tests";
}

static void mock_dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free  = 1ull << 30;
    *total = 1ull << 30;
}

static enum ggml_backend_dev_type mock_dev_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void mock_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = mock_dev_get_name(dev);
    props->description = mock_dev_get_description(dev);
    mock_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type      = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = nullptr;
    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ true,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t mock_dev_init_backend(ggml_backend_dev_t dev, const char *) {
    mock_stream * s = new mock_stream;
    s->dev = dev;
    s->cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(s->cpu, 2);
    s->worker = std::thread([s] { s->run(); });
    dev_ctx(dev)->n_instances++;
    return new ggml_backend {
        /* .guid    = */ mock_guid(),
        /* .iface   = */ mock_backend_iface,
        /* .device  = */ dev,
        /* .context = */ s,
    };
}

static ggml_backend_buffer_type_t mock_dev_get_buffer_type(ggml_backend_dev_t dev) {
    return &dev_ctx(dev)->buft;
}

static bool mock_dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    // SCALE stays on the CPU, so tests can put a CPU split between two device splits
    if (op->op == GGML_OP_SCALE) {
        return false;
    }
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    return ggml_backend_dev_supports_op(cpu, op);
}

static bool mock_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == &dev_ctx(dev)->buft;
}

static bool mock_dev_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    const int64_t batch = op->op == GGML_OP_MUL_MAT_ID ? op->ne[2] : op->ne[1];
    return batch >= dev_ctx(dev)->offload_min_batch;
}

static ggml_backend_event_t mock_dev_event_new(ggml_backend_dev_t dev) {
    return new ggml_backend_event { dev, new mock_event };
}

static void mock_dev_event_free(ggml_backend_dev_t, ggml_backend_event_t event) {
    if (event == nullptr) {
        return;
    }
    delete (mock_event *) event->context;
    delete event;
}

static void mock_dev_event_synchronize(ggml_backend_dev_t, ggml_backend_event_t event) {
    mock_event * ev = (mock_event *) event->context;
    std::unique_lock<std::mutex> lock(ev->m);
    const uint64_t target = ev->recorded;
    ev->cv.wait(lock, [&] { return ev->done >= target; });
}

static const ggml_backend_device_i mock_dev_iface = {
    /* .get_name             = */ mock_dev_get_name,
    /* .get_description      = */ mock_dev_get_description,
    /* .get_memory           = */ mock_dev_get_memory,
    /* .get_type             = */ mock_dev_get_type,
    /* .get_props            = */ mock_dev_get_props,
    /* .init_backend         = */ mock_dev_init_backend,
    /* .get_buffer_type      = */ mock_dev_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ mock_dev_supports_op,
    /* .supports_buft        = */ mock_dev_supports_buft,
    /* .offload_op           = */ mock_dev_offload_op,
    /* .event_new            = */ mock_dev_event_new,
    /* .event_free           = */ mock_dev_event_free,
    /* .event_synchronize    = */ mock_dev_event_synchronize,
};

//
// registry, only so that ggml_backend_dev_backend_reg has something to return
//

static const char * mock_reg_get_name(ggml_backend_reg_t) {
    return "MOCK";
}

static size_t mock_reg_get_device_count(ggml_backend_reg_t) {
    return 1;
}

static ggml_backend_dev_t mock_reg_get_device(ggml_backend_reg_t reg, size_t) {
    return (ggml_backend_dev_t) reg->context;
}

static const ggml_backend_reg_i mock_reg_iface = {
    /* .get_name         = */ mock_reg_get_name,
    /* .get_device_count = */ mock_reg_get_device_count,
    /* .get_device       = */ mock_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_dev_t mock_gpu_dev_new(int64_t copy_delay_us, int64_t offload_min_batch) {
    mock_dev_ctx * ctx = new mock_dev_ctx;
    ctx->copy_delay_us     = copy_delay_us;
    ctx->offload_min_batch = offload_min_batch;

    ggml_backend_dev_t dev = new ggml_backend_device {
        /* .iface   = */ mock_dev_iface,
        /* .reg     = */ &ctx->reg,
        /* .context = */ ctx,
    };
    ctx->buft = { mock_buft_iface, dev, nullptr };
    ctx->reg  = { GGML_BACKEND_API_VERSION, mock_reg_iface, dev };
    return dev;
}

void mock_gpu_dev_free(ggml_backend_dev_t dev) {
    delete dev_ctx(dev);
    delete dev;
}

mock_gpu_stats mock_gpu_get_stats(ggml_backend_dev_t dev) {
    mock_dev_ctx * d = dev_ctx(dev);
    mock_gpu_stats st;
    st.n_set_async  = d->n_set_async;
    st.n_graphs     = d->n_graphs;
    st.n_instances  = d->n_instances;
    st.n_overlap_us = d->n_overlap_us;
    return st;
}

void mock_gpu_reset_stats(ggml_backend_dev_t dev) {
    mock_dev_ctx * d = dev_ctx(dev);
    d->n_set_async  = 0;
    d->n_graphs     = 0;
    d->n_instances  = 0;
    d->n_overlap_us = 0;
}
