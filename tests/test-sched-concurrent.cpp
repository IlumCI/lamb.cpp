// Tests for concurrent host and device splits in ggml_backend_sched (GGML_SCHED_CONCURRENT=1).
// The graph has the shape of a MoE layer with a shared expert or an expert cache: device work, then host work, then
// device work that does not read the host result, then a node that joins both. With the flag, the independent device
// work must run while the host works; without it, it must not; the results must be bit-identical to a CPU-only run.

#include "mock-gpu-backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

constexpr int64_t K = 64;
constexpr int64_t T = 8;

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

// the host work: doubles its input slowly, and records whether the device was busy meanwhile
static ggml_backend_dev_t g_dev       = nullptr;
static std::atomic<bool>  g_overlap   {false};
static std::atomic<int>   g_host_runs {0};

static void host_work(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void *) {
    if (ith == 0) {
        g_host_runs++;
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(60)) {
            if (g_dev && mock_gpu_running_graphs(g_dev) > 0) {
                g_overlap = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        for (int64_t i = 0; i < ggml_nelements(dst); i++) {
            ((float *) dst->data)[i] = 2.0f*((const float *) a->data)[i];
        }
    }
    (void) nth;
}

struct weights {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         w0  = nullptr;
    ggml_tensor *         w1  = nullptr;

    weights(ggml_backend_buffer_type_t buft) {
        ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        w0  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, K);
        w1  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, K);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::mt19937 rng(5);
        std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
        for (ggml_tensor * t : {w0, w1}) {
            std::vector<float> v(ggml_nelements(t));
            for (float & x : v) {
                x = dist(rng);
            }
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
    }
    ~weights() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

static std::vector<float> run(ggml_backend_sched_t sched, const weights & wt, int seed) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_set_input(x);
    ggml_tensor * a   = ggml_mul_mat(ctx, wt.w0, x);                   // device
    ggml_tensor * h   = ggml_map_custom1(ctx, a, host_work, 1, nullptr); // host
    ggml_tensor * g   = ggml_tanh(ctx, ggml_mul_mat(ctx, wt.w1, a));   // device, independent of h
    ggml_tensor * out = ggml_add(ctx, g, h);                           // joins both
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, h); // host work first in the graph, as build_moe_ffn orders it
    ggml_build_forward_expand(gf, out);

    std::vector<float> ret(ggml_nelements(out));
    ggml_backend_sched_reset(sched);
    if (ggml_backend_sched_alloc_graph(sched, gf)) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> xv(ggml_nelements(x));
        for (float & v : xv) {
            v = dist(rng);
        }
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        CHECK(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, ret.data(), 0, ggml_nbytes(out));
    } else {
        CHECK(false && "alloc_graph failed");
    }
    ggml_free(ctx);
    return ret;
}

static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

// returns whether the device ran during the host work, and the results
static bool run_device(bool concurrent, std::vector<std::vector<float>> & outs, int n_runs, int * n_splits) {
    set_env("GGML_SCHED_CONCURRENT", concurrent ? "1" : nullptr);
    ggml_backend_dev_t dev = mock_gpu_dev_new(0, 1);
    mock_gpu_set_compute_delay(dev, 20000);
    g_dev = dev;
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 2);
    ggml_backend_t backends[2] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 256, false, true);
    set_env("GGML_SCHED_CONCURRENT", nullptr);

    bool overlap = false;
    {
        weights wt(ggml_backend_dev_buffer_type(dev));
        for (int r = 0; r < n_runs; r++) {
            g_overlap = false;
            outs.push_back(run(sched, wt, 10 + r));
            overlap = overlap || g_overlap;
        }
        *n_splits = ggml_backend_sched_get_n_splits(sched);
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    g_dev = nullptr;
    mock_gpu_dev_free(dev);
    return overlap;
}

int main() {
    ggml_backend_load_all();
    const int n_runs = 3;

    std::vector<std::vector<float>> ref;
    {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(cpu, 2);
        ggml_backend_sched_t sched = ggml_backend_sched_new(&cpu, nullptr, 1, 256, false, true);
        weights wt(ggml_backend_cpu_buffer_type());
        for (int r = 0; r < n_runs; r++) {
            ref.push_back(run(sched, wt, 10 + r));
        }
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
    }

    std::vector<std::vector<float>> seq;
    std::vector<std::vector<float>> conc;
    int splits_seq  = 0;
    int splits_conc = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const bool overlap_seq  = run_device(false, seq,  n_runs, &splits_seq);
    const auto t1 = std::chrono::steady_clock::now();
    const bool overlap_conc = run_device(true,  conc, n_runs, &splits_conc);
    const auto t2 = std::chrono::steady_clock::now();

    const double ms_seq  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_conc = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("sequential: %d splits, overlap %d, %.0f ms\n", splits_seq,  (int) overlap_seq,  ms_seq);
    printf("concurrent: %d splits, overlap %d, %.0f ms\n", splits_conc, (int) overlap_conc, ms_conc);

    CHECK(splits_seq >= 3);
    CHECK(!overlap_seq);
    CHECK(overlap_conc);
    // per run: 20 + 60 + 60 ms in sequence, 20 + max(60, 40) + 20 ms concurrently
    CHECK(ms_conc < 0.9*ms_seq);
    for (int r = 0; r < n_runs; r++) {
        CHECK(same(seq[r],  ref[r]));
        CHECK(same(conc[r], ref[r]));
    }
    CHECK(g_host_runs == 3*n_runs);

    if (n_fail > 0) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
