// Tests for weight prefetch in ggml_backend_sched (GGML_SCHED_PREFETCH=1).
// A mock asynchronous GPU computes splits whose weights live in host memory; with prefetch the weights are uploaded
// on a second stream while the previous split runs. The results must be bit-identical to a CPU-only run.

#include "mock-gpu-backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

constexpr int64_t K        = 128; // width of the hidden state
constexpr int64_t N_TOK    = 64;  // tokens per batch, above the offload threshold of the mock
constexpr int     N_LAYER  = 8;
constexpr int64_t N_EXPERT = 16;
constexpr int64_t N_USED   = 4;
constexpr int     N_THREAD = 2;

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// weights in a host buffer marked as weights, as llama.cpp does for weights it keeps in system memory
struct weights {
    ggml_context *              ctx = nullptr;
    ggml_backend_buffer_t       buf = nullptr;
    std::vector<ggml_tensor *>  w;
    ggml_tensor *               we  = nullptr; // expert stack

    weights() {
        ggml_init_params ip = { (N_LAYER + 2)*ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        for (int il = 0; il < N_LAYER; il++) {
            // one quantized layer, so that copies of block data are covered too
            const ggml_type type = il == 3 ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
            w.push_back(ggml_new_tensor_2d(ctx, type, K, K));
        }
        we  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, K, N_EXPERT);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        const float scale = 1.5f/std::sqrt((float) K);
        for (ggml_tensor * t : w) {
            std::vector<float> v(ggml_nelements(t));
            for (float & x : v) {
                x = dist(rng)*scale;
            }
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
            } else {
                std::vector<uint8_t> q(ggml_nbytes(t));
                ggml_quantize_chunk(t->type, v.data(), q.data(), 0, t->ne[1], t->ne[0], nullptr);
                ggml_backend_tensor_set(t, q.data(), 0, q.size());
            }
        }
        std::vector<float> v(ggml_nelements(we));
        for (float & x : v) {
            x = dist(rng)*scale;
        }
        ggml_backend_tensor_set(we, v.data(), 0, ggml_nbytes(we));
    }

    ~weights() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

struct inputs {
    std::vector<float>   x;
    std::vector<int32_t> ids;
};

static inputs make_inputs(int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int32_t> expert(0, N_EXPERT - 1);
    inputs in;
    in.x.resize(K*N_TOK);
    for (float & v : in.x) {
        v = dist(rng);
    }
    in.ids.resize(N_USED*N_TOK);
    for (int64_t t = 0; t < N_TOK; t++) {
        // distinct experts per token
        std::vector<int32_t> pick;
        while ((int64_t) pick.size() < N_USED) {
            const int32_t e = expert(rng);
            bool dup = false;
            for (int32_t p : pick) {
                dup = dup || p == e;
            }
            if (!dup) {
                pick.push_back(e);
            }
        }
        for (int64_t u = 0; u < N_USED; u++) {
            in.ids[t*N_USED + u] = pick[u];
        }
    }
    return in;
}

// layers of matmul + tanh; a SCALE every third layer runs on the CPU; one MoE layer; the first weight is used twice
static std::vector<float> run(ggml_backend_sched_t sched, const weights & wt, const inputs & in) {
    ggml_init_params ip = { 256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N_TOK);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, N_TOK);
    ggml_set_input(x);
    ggml_set_input(ids);

    ggml_tensor * y = x;
    for (int il = 0; il < N_LAYER; il++) {
        y = ggml_tanh(ctx, ggml_mul_mat(ctx, wt.w[il], y));
        if (il % 3 == 1) {
            y = ggml_scale(ctx, y, 0.9f);
        }
        if (il == 4) {
            ggml_tensor * e = ggml_mul_mat_id(ctx, wt.we, ggml_reshape_3d(ctx, y, K, 1, N_TOK), ids);
            y = ggml_cont(ctx, ggml_view_2d(ctx, e, K, N_TOK, e->nb[2], 0));
        }
    }
    y = ggml_tanh(ctx, ggml_mul_mat(ctx, wt.w[0], y));
    ggml_set_output(y);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_sched_reset(sched);
    std::vector<float> out;
    if (ggml_backend_sched_alloc_graph(sched, gf)) {
        ggml_backend_tensor_set(x,   in.x.data(),   0, ggml_nbytes(x));
        ggml_backend_tensor_set(ids, in.ids.data(), 0, ggml_nbytes(ids));
        CHECK(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
        out.resize(ggml_nelements(y));
        ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    } else {
        CHECK(false && "alloc_graph failed");
    }
    ggml_free(ctx);
    return out;
}

static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !a.empty() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

static bool finite(const std::vector<float> & a) {
    for (float v : a) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

struct setup {
    ggml_backend_dev_t   dev;
    ggml_backend_t       gpu;
    ggml_backend_t       cpu;
    ggml_backend_sched_t sched;

    setup(int64_t copy_delay_us, bool prefetch) {
        set_env("GGML_SCHED_PREFETCH", prefetch ? "1" : nullptr);
        dev = mock_gpu_dev_new(copy_delay_us, 32);
        gpu = ggml_backend_dev_init(dev, nullptr);
        cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(cpu, N_THREAD);
        ggml_backend_t backends[2] = { gpu, cpu };
        sched = ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true);
        set_env("GGML_SCHED_PREFETCH", nullptr);
    }

    ~setup() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
        ggml_backend_free(gpu);
        mock_gpu_dev_free(dev);
    }
};

static std::vector<std::vector<float>> reference(const weights & wt, int n_runs) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, N_THREAD);
    ggml_backend_sched_t sched = ggml_backend_sched_new(&cpu, nullptr, 1, 2048, false, true);
    std::vector<std::vector<float>> ret;
    for (int r = 0; r < n_runs; r++) {
        ret.push_back(run(sched, wt, make_inputs(100 + r)));
        CHECK(finite(ret.back()));
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    return ret;
}

int main() {
    ggml_backend_load_all();

    const int n_runs = 4;
    weights wt;
    const auto ref = reference(wt, n_runs);

    // without prefetch: the device path must match the CPU, and nothing overlaps
    size_t size_plain = 0;
    {
        setup s(300, false);
        for (int r = 0; r < n_runs; r++) {
            CHECK(same(run(s.sched, wt, make_inputs(100 + r)), ref[r]));
        }
        const mock_gpu_stats st = mock_gpu_get_stats(s.dev);
        CHECK(st.n_instances == 1);
        CHECK(st.n_overlap_us == 0);
        CHECK(ggml_backend_sched_get_n_splits(s.sched) > N_LAYER);
        size_plain = ggml_backend_sched_get_buffer_size(s.sched, s.gpu);
        printf("no prefetch:   %d splits, %lld graphs, %lld async uploads, %lld us of overlap\n",
            ggml_backend_sched_get_n_splits(s.sched), (long long) st.n_graphs, (long long) st.n_set_async, (long long) st.n_overlap_us);
    }

    // with prefetch and slow copies: same results, and the uploads overlap device compute
    {
        setup s(300, true);
        for (int r = 0; r < n_runs; r++) {
            CHECK(same(run(s.sched, wt, make_inputs(100 + r)), ref[r]));
        }
        const mock_gpu_stats st = mock_gpu_get_stats(s.dev);
        CHECK(st.n_instances == 2); // the compute stream and the copy stream
        CHECK(st.n_set_async > n_runs*(N_LAYER - 3));
        CHECK(st.n_overlap_us > 0);
        // the staging slots are reported as compute memory
        CHECK(ggml_backend_sched_get_buffer_size(s.sched, s.gpu) > size_plain);
        printf("prefetch:      %d splits, %lld graphs, %lld async uploads, %lld us of overlap, staging +%zu bytes\n",
            ggml_backend_sched_get_n_splits(s.sched), (long long) st.n_graphs, (long long) st.n_set_async, (long long) st.n_overlap_us,
            ggml_backend_sched_get_buffer_size(s.sched, s.gpu) - size_plain);
    }

    // with prefetch and no delay, many runs: a missing wait shows up as a different result
    {
        setup s(0, true);
        for (int r = 0; r < 50; r++) {
            CHECK(same(run(s.sched, wt, make_inputs(100 + r % n_runs)), ref[r % n_runs]));
        }
    }

    if (n_fail > 0) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
