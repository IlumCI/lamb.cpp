// Tests for the cost based op offload decision (ggml_backend_dev_set_offload_params).

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

// a PCIe 4.0 GPU next to a desktop CPU, the same numbers for every type
static ggml_backend_offload_params mock_params() {
    ggml_backend_offload_params p = {};
    p.h2d_gbps = 25.0f;
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        p.host_gemv_gbps  [t] = 60.0f;
        p.host_gemm_tflops[t] = 1.0f;
        p.dev_gemv_gbps   [t] = 450.0f;
        p.dev_gemm_tflops [t] = 60.0f;
    }
    return p;
}

struct graph_ctx {
    ggml_context * ctx;
    graph_ctx() {
        ggml_init_params ip = { 64*ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);
    }
    ~graph_ctx() { ggml_free(ctx); }
};

// dense [4096, 4096] q4_K weight times a batch of n_tokens
static ggml_tensor * dense_op(ggml_context * ctx, int64_t n_tokens) {
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 4096, 4096);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, n_tokens);
    return ggml_mul_mat(ctx, w, x);
}

// 128 experts of [2048, 768] q4_K, 8 used per token
static ggml_tensor * moe_op(ggml_context * ctx, int64_t n_tokens) {
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, 2048, 768, 128);
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 8, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, n_tokens);
    return ggml_mul_mat_id(ctx, w, x, ids);
}

// smallest batch that is offloaded, or -1
static int64_t threshold(ggml_backend_dev_t dev, ggml_tensor * (*make)(ggml_context *, int64_t)) {
    for (int64_t n = 1; n <= 8192; n *= 2) {
        graph_ctx g;
        if (ggml_backend_dev_offload_op(dev, make(g.ctx, n))) {
            return n;
        }
    }
    return -1;
}

static void test_cost() {
    const ggml_backend_offload_params p = mock_params();
    double th = 0.0;
    double td = 0.0;

    // one token over a 64 MiB weight: the host reads it once, the GPU must copy it first
    CHECK(ggml_backend_offload_cost(&p, GGML_TYPE_Q4_K, 64e6, 1.0, 2.0*64e6, &th, &td));
    CHECK(th > 0.0 && td > th);
    const double td_full = td;

    // a large batch is limited by arithmetic on the host
    CHECK(ggml_backend_offload_cost(&p, GGML_TYPE_Q4_K, 64e6, 1.0, 2.0*64e6*4096, &th, &td));
    CHECK(td < th);

    // reading half the weight halves the copy
    double th2 = 0.0;
    double td2 = 0.0;
    CHECK(ggml_backend_offload_cost(&p, GGML_TYPE_Q4_K, 64e6, 0.5, 2.0*64e6, &th2, &td2));
    CHECK(td2 < td_full);
    CHECK(std::fabs(th2 - 0.5*64e6/60e9) < 1e-9);

    // missing numbers give no decision
    ggml_backend_offload_params q = p;
    q.h2d_gbps = 0.0f;
    CHECK(!ggml_backend_offload_cost(&q, GGML_TYPE_Q4_K, 64e6, 1.0, 1e9, &th, &td));
    q = p;
    q.host_gemm_tflops[GGML_TYPE_Q8_0] = 0.0f;
    CHECK(!ggml_backend_offload_cost(&q, GGML_TYPE_Q8_0, 64e6, 1.0, 1e9, &th, &td));
    CHECK( ggml_backend_offload_cost(&q, GGML_TYPE_Q4_K, 64e6, 1.0, 1e9, &th, &td));
}

static void test_policy(ggml_backend_dev_t dev) {
    // without params the backend rule decides, the CPU device never offloads
    CHECK(threshold(dev, dense_op) == -1);
    CHECK(threshold(dev, moe_op)   == -1);

    const ggml_backend_offload_params p = mock_params();
    ggml_backend_dev_set_offload_params(dev, &p);

    const int64_t th_dense = threshold(dev, dense_op);
    const int64_t th_moe   = threshold(dev, moe_op);
    printf("offload from batch: dense %lld, moe %lld\n", (long long) th_dense, (long long) th_moe);

    CHECK(th_dense > 1);
    CHECK(th_moe   > 1);
    // a MoE stack is offloaded later: a small batch reads few experts on the host but copies most of them
    CHECK(th_moe > th_dense);
    // the fixed rule offloads both at 32, which is too early for this MoE
    CHECK(th_moe > 32);

    // the decision is monotonic in the batch size
    bool seen = false;
    for (int64_t n = 1; n <= 8192; n *= 2) {
        graph_ctx g;
        const bool off = ggml_backend_dev_offload_op(dev, moe_op(g.ctx, n));
        CHECK(!(seen && !off));
        seen = seen || off;
    }

    // other ops keep the backend rule
    {
        graph_ctx g;
        ggml_tensor * a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 4096, 4096);
        CHECK(!ggml_backend_dev_offload_op(dev, ggml_add(g.ctx, a, a)));
    }

    // GGML_OP_OFFLOAD_MIN_BATCH in the environment restores the backend rule
#ifdef _WIN32
    _putenv_s("GGML_OP_OFFLOAD_MIN_BATCH", "32");
#else
    setenv("GGML_OP_OFFLOAD_MIN_BATCH", "32", 1);
#endif
    CHECK(threshold(dev, dense_op) == -1);
#ifdef _WIN32
    _putenv_s("GGML_OP_OFFLOAD_MIN_BATCH", "");
#else
    unsetenv("GGML_OP_OFFLOAD_MIN_BATCH");
#endif
    CHECK(threshold(dev, dense_op) == th_dense);

    // a slower link moves the threshold up
    ggml_backend_offload_params slow = p;
    slow.h2d_gbps = 5.0f;
    ggml_backend_dev_set_offload_params(dev, &slow);
    CHECK(threshold(dev, dense_op) > th_dense);

    // clearing restores the backend rule
    ggml_backend_dev_set_offload_params(dev, nullptr);
    CHECK(threshold(dev, dense_op) == -1);
}

int main() {
    ggml_backend_load_all();

    test_cost();

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    CHECK(dev != nullptr);
    if (dev) {
        test_policy(dev);
    }

    if (n_fail > 0) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
