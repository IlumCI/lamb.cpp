// Tests for MUL_MAT_ID (MoE experts) with weights in the AMX buffer type of the CPU backend.
// The result must match the same op with the weights in a plain CPU buffer; with --perf the two are also timed
// on a prefill-sized and a decode-sized batch. Skips when the CPU has no AMX.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

static ggml_backend_buffer_type_t find_extra_buft(const char * name) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto fn = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    for (ggml_backend_buffer_type_t * b = fn ? fn(dev) : nullptr; b && *b; ++b) {
        if (strcmp(ggml_backend_buft_name(*b), name) == 0) {
            return *b;
        }
    }
    return nullptr;
}

struct mmid_case {
    ggml_type type;
    int64_t   K, N, n_as, n_used, n_tok;
    bool      broadcast; // b has one row per token, shared by its experts
};

struct runner {
    ggml_context *        ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_tensor *         as    = nullptr;

    // the expert weights in buft
    runner(const mmid_case & c, ggml_backend_buffer_type_t buft, const std::vector<uint8_t> & q) {
        ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
        ctx_w = ggml_init(ip);
        as    = ggml_new_tensor_3d(ctx_w, c.type, c.K, c.N, c.n_as);
        buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
        ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // an extra buffer type without a layout for this tensor leaves extra unset, the loader would not pick it
        ok = buft == ggml_backend_cpu_buffer_type() || as->extra != nullptr;
        if (ok) {
            ggml_backend_tensor_set(as, q.data(), 0, q.size());
        }
    }

    bool ok = false;
    ~runner() {
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx_w);
    }

    // run the op n_rep times, returns the output and the best time in us
    std::vector<float> run(ggml_backend_t backend, const mmid_case & c, const std::vector<float> & bv, const std::vector<int32_t> & idv, int n_rep, double * best_us) {
        ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.broadcast ? 1 : c.n_used, c.n_tok);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.n_tok);
        ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(b, bv.data(), 0, ggml_nbytes(b));
        ggml_backend_tensor_set(ids, idv.data(), 0, ggml_nbytes(ids));
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        double best = 1e30;
        for (int r = 0; r < n_rep; r++) {
            const auto t0 = std::chrono::steady_clock::now();
            CHECK(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
            best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        if (best_us) {
            *best_us = best;
        }
        std::vector<float> ret(ggml_nelements(out));
        ggml_backend_tensor_get(out, ret.data(), 0, ggml_nbytes(out));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return ret;
    }
};

static double nmse(const std::vector<float> & a, const std::vector<float> & ref) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i])) {
            return INFINITY;
        }
        num += (a[i] - ref[i])*(a[i] - ref[i]);
        den += ref[i]*ref[i];
    }
    return den > 0.0 ? num/den : num;
}

static void test_case(ggml_backend_t backend, ggml_backend_buffer_type_t amx, const mmid_case & c, int n_rep, bool print_perf) {
    std::mt19937 rng(c.type*131 + c.n_tok);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> wf(c.K*c.N*c.n_as);
    for (float & v : wf) {
        v = dist(rng);
    }
    std::vector<uint8_t> q(ggml_row_size(c.type, c.K)*c.N*c.n_as);
    ggml_quantize_chunk(c.type, wf.data(), q.data(), 0, c.N*c.n_as, c.K, nullptr);

    std::vector<float> bv(c.K*(c.broadcast ? 1 : c.n_used)*c.n_tok);
    for (float & v : bv) {
        v = dist(rng);
    }
    // distinct experts per token, skewed so that some experts get many rows and some one
    std::vector<int32_t> idv(c.n_used*c.n_tok);
    std::uniform_int_distribution<int32_t> pick(0, (int32_t) c.n_as - 1);
    for (int64_t t = 0; t < c.n_tok; t++) {
        for (int64_t j = 0; j < c.n_used; j++) {
            int32_t e;
            bool dup;
            do {
                e = (t % 3 == 0 && j == 0) ? 0 : pick(rng);
                dup = false;
                for (int64_t k = 0; k < j; k++) {
                    dup = dup || idv[t*c.n_used + k] == e;
                }
            } while (dup);
            idv[t*c.n_used + j] = e;
        }
    }

    double t_ref = 0.0;
    double t_amx = 0.0;
    double t_rpk = 0.0;
    runner ref(c, ggml_backend_cpu_buffer_type(), q);
    runner amx_r(c, amx, q);
    CHECK(amx_r.ok);
    const auto out_ref = ref.run(backend, c, bv, idv, n_rep, &t_ref);
    const auto out_amx = amx_r.run(backend, c, bv, idv, n_rep, &t_amx);
    const double err = nmse(out_amx, out_ref);
    CHECK(err < 1e-5);

    // the repacked CPU kernels are what experts in system memory use with --load-mode none
    ggml_backend_buffer_type_t rpk = print_perf ? find_extra_buft("CPU_REPACK") : nullptr;
    if (rpk) {
        runner rpk_r(c, rpk, q);
        if (rpk_r.ok) {
            rpk_r.run(backend, c, bv, idv, n_rep, &t_rpk);
        }
    }

    if (print_perf || err >= 1e-5) {
        printf("%-6s K %5lld N %5lld experts %3lld used %lld tokens %4lld%s: nmse %.2e, cpu %9.1f us, repack %9.1f us, amx %9.1f us, %.2fx over the faster\n",
            ggml_type_name(c.type), (long long) c.K, (long long) c.N, (long long) c.n_as, (long long) c.n_used, (long long) c.n_tok,
            c.broadcast ? " bcast" : "", err, t_ref, t_rpk, t_amx, std::min(t_ref, t_rpk > 0.0 ? t_rpk : t_ref)/t_amx);
    }
}

int main(int argc, char ** argv) {
    const bool perf = argc > 1 && std::string(argv[1]) == "--perf";

    ggml_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) {
            fputs(text, stderr);
        }
    }, nullptr);
    ggml_backend_load_all();
    ggml_backend_buffer_type_t amx = find_extra_buft("AMX");
    if (amx == nullptr) {
        printf("no AMX buffer type on this CPU, skipping\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 4);

    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS };
    for (ggml_type type : types) {
        for (int64_t n_tok : {1, 5, 40}) {
            for (bool bcast : {true, false}) {
                test_case(backend, amx, { type, 256, 64, 16, 4, n_tok, bcast }, 1, false);
            }
        }
    }

    if (perf) {
        // a 30B-A3B-sized layer: 128 experts of 2048 -> 768, 8 used
        for (ggml_type type : { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS }) {
            for (int64_t n_tok : {1, 32, 512}) {
                test_case(backend, amx, { type, 2048, 768, 128, 8, n_tok, true }, 5, true);
            }
        }
    }

    ggml_backend_free(backend);

    if (n_fail > 0) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
