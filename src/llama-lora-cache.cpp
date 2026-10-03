#include "llama-lora-cache.h"

#include "llama-adapter.h"
#include "llama-impl.h"

#include <algorithm>

llama_lora_cache::llama_lora_cache(size_t budget, bool allow_host) : budget_(budget), allow_host_(allow_host) {
    worker = std::thread([this] { run(); });
}

llama_lora_cache::~llama_lora_cache() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
        queue.clear();
    }
    cv.notify_all();
    worker.join();
}

llama_lora_cache::entry & llama_lora_cache::get_entry(llama_adapter_lora * adapter, ggml_backend_buffer_type_t buft) {
    for (entry & e : entries) {
        if (e.adapter == adapter && e.buft == buft) {
            return e;
        }
    }
    entries.emplace_back();
    entry & e = entries.back();
    e.adapter = adapter;
    e.buft    = buft;

    const size_t align = ggml_backend_buft_get_alignment(buft);
    for (const std::string & name : adapter->tiered.at(buft)) {
        const llama_adapter_lora_weight & w = adapter->ab_map.at(name);
        e.bytes += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, w.a_host), align);
        e.bytes += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, w.b_host), align);
    }
    return e;
}

int llama_lora_cache::n_users(const llama_adapter_lora * adapter) const {
    int n = 0;
    for (const auto & [user, set] : users) {
        n += std::count(set.begin(), set.end(), adapter) > 0;
    }
    return n;
}

void llama_lora_cache::point(entry & e, bool to_device) {
    for (const auto & d : e.dev) {
        d.w->a = to_device ? d.a : d.w->a_host;
        d.w->b = to_device ? d.b : d.w->b_host;
    }
}

bool llama_lora_cache::make_room(ggml_backend_buffer_type_t buft, size_t bytes) {
    if (bytes > budget_) {
        return false;
    }
    while (used[buft] + bytes > budget_) {
        // the least recently used copy that no user applies
        entry * victim = nullptr;
        for (entry & e : entries) {
            if (e.buft == buft && e.state == entry::RESIDENT && n_users(e.adapter) == 0 &&
                (victim == nullptr || e.last_use < victim->last_use)) {
                victim = &e;
            }
        }
        if (victim == nullptr) {
            return false;
        }
        point(*victim, false);
        victim->dev.clear();
        victim->buf.reset();
        victim->ctx.reset();
        victim->state = entry::HOST;
        used[buft] -= victim->bytes;
        info_.n_evictions++;
        generation++;
    }
    return true;
}

void llama_lora_cache::use(const void * user, const std::vector<llama_adapter_lora *> & adapters) {
    std::lock_guard<std::mutex> lock(mutex);
    users[user] = adapters;
    if (adapters.empty()) {
        users.erase(user);
    }
    for (llama_adapter_lora * adapter : adapters) {
        for (const auto & [buft, names] : adapter->tiered) {
            entry & e = get_entry(adapter, buft);
            e.last_use = ++clock;
            if (e.state != entry::HOST || e.failed) {
                continue;
            }
            if (!make_room(buft, e.bytes)) {
                // it stays in system memory, its matmuls run on the host
                continue;
            }
            used[buft] += e.bytes;
            e.state = entry::UPLOADING;
            queue.push_back(&e);
        }
    }
    cv.notify_one();
}

uint64_t llama_lora_cache::poll() {
    std::lock_guard<std::mutex> lock(mutex);
    for (entry & e : entries) {
        if (e.state != entry::DONE) {
            continue;
        }
        if (e.failed) {
            e.state = entry::HOST;
            used[e.buft] -= e.bytes;
            info_.n_failed++;
            continue;
        }
        e.state = entry::RESIDENT;
        point(e, true);
        info_.n_uploads++;
        generation++;
    }
    return generation;
}

void llama_lora_cache::forget(llama_adapter_lora * adapter) {
    std::unique_lock<std::mutex> lock(mutex);
    // an upload of this adapter may be in flight
    cv_idle.wait(lock, [&] {
        if (busy) {
            return false;
        }
        for (entry * e : queue) {
            if (e->adapter == adapter) {
                return false;
            }
        }
        return true;
    });
    for (auto it = entries.begin(); it != entries.end();) {
        if (it->adapter != adapter) {
            ++it;
            continue;
        }
        if (it->state != entry::HOST) {
            used[it->buft] -= it->bytes;
        }
        point(*it, false);
        it = entries.erase(it);
    }
    for (auto & [user, set] : users) {
        set.erase(std::remove(set.begin(), set.end(), adapter), set.end());
    }
    generation++;
}

llama_lora_cache::info llama_lora_cache::get_info() {
    std::lock_guard<std::mutex> lock(mutex);
    info ret = info_;
    ret.n_bytes    = 0;
    ret.n_resident = 0;
    for (const auto & [buft, n] : used) {
        ret.n_bytes += n;
    }
    for (const entry & e : entries) {
        ret.n_resident += e.state == entry::RESIDENT;
    }
    return ret;
}

void llama_lora_cache::run() {
    for (;;) {
        entry * e = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return stop || !queue.empty(); });
            if (stop) {
                return;
            }
            e = queue.front();
            queue.pop_front();
            busy = true;
        }

        // only this thread touches the entry until its state is DONE
        const std::vector<std::string> & names = e->adapter->tiered.at(e->buft);
        ggml_init_params params = {
            /*.mem_size   =*/ 2*names.size()*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        bool ok = false;
        ggml_context * ctx = ggml_init(params);
        if (ctx != nullptr) {
            e->ctx.reset(ctx);
            for (const std::string & name : names) {
                llama_adapter_lora_weight & w = e->adapter->ab_map.at(name);
                ggml_tensor * a = ggml_dup_tensor(ctx, w.a_host);
                ggml_tensor * b = ggml_dup_tensor(ctx, w.b_host);
                ggml_set_name(a, w.a_host->name);
                ggml_set_name(b, w.b_host->name);
                e->dev.push_back({ &w, a, b });
            }
            e->buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, e->buft));
            if (e->buf) {
                for (const auto & d : e->dev) {
                    ggml_backend_tensor_set(d.a, d.w->a_host->data, 0, ggml_nbytes(d.a));
                    ggml_backend_tensor_set(d.b, d.w->b_host->data, 0, ggml_nbytes(d.b));
                }
                ok = true;
            }
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: failed to allocate %zu bytes of %s for a LoRA adapter, it stays in system memory\n",
                __func__, e->bytes, ggml_backend_buft_name(e->buft));
            e->dev.clear();
            e->buf.reset();
            e->ctx.reset();
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            e->failed = !ok;
            e->state  = entry::DONE;
            busy = false;
        }
        cv_idle.notify_all();
    }
}
