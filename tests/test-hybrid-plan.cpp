// Tests for the hybrid placement cost model.
// The synthetic part needs no model; with --models DIR every generated test model is also loaded with no_alloc and estimated.

#include "hybrid-plan.h"

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

// a 12 GB PCIe 4.0 GPU next to a desktop CPU with dual-channel DDR5
static common_hw_profile mock_profile() {
    common_hw_profile p;
    p.key = "mock";

    common_hw_dev_profile cpu;
    cpu.name = "CPU";
    cpu.type = GGML_BACKEND_DEVICE_TYPE_CPU;
    cpu.gemv_gbps["q4_K"]   = 60.0;
    cpu.gemv_gbps["f16"]    = 70.0;
    cpu.gemm_tflops["q4_K"] = 1.0;
    cpu.gemm_tflops["f16"]  = 0.8;

    common_hw_dev_profile gpu;
    gpu.name = "GPU0";
    gpu.type = GGML_BACKEND_DEVICE_TYPE_GPU;
    gpu.h2d_gbps = 25.0;
    gpu.d2h_gbps = 25.0;
    gpu.gemv_gbps["q4_K"]   = 450.0;
    gpu.gemv_gbps["f16"]    = 480.0;
    gpu.gemm_tflops["q4_K"] = 60.0;
    gpu.gemm_tflops["f16"]  = 80.0;

    p.devs = {cpu, gpu};
    return p;
}

static hp_tensor make_tensor(const std::string & name, size_t n_params, ggml_type type, int32_t dev) {
    hp_tensor t;
    t.name     = name;
    t.layer    = hp_layer_of(name);
    t.type     = type;
    t.n_params = n_params;
    t.bytes    = n_params*ggml_type_size(type)/ggml_blck_size(type);
    t.use      = hp_classify(name);
    if (t.use == HP_USE_ROWS) {
        t.n_rows = 150000;
    }
    t.dev = dev;
    return t;
}

// layer shapes roughly of an 8B dense or a 30B-A3B MoE model in q4_K
static hp_model make_model(bool moe, int32_t n_layer) {
    hp_model m;
    m.n_layer   = n_layer;
    m.dev_names = {"GPU0"};
    m.ctx_bytes = {0, 0};
    if (moe) {
        m.n_expert      = 128;
        m.n_expert_used = 8;
    }
    m.tensors.push_back(make_tensor("token_embd.weight", 600000000, GGML_TYPE_Q4_K, -1));
    for (int32_t il = 0; il < n_layer; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        m.tensors.push_back(make_tensor(p + "attn_norm.weight", 4096, GGML_TYPE_F32, -1));
        m.tensors.push_back(make_tensor(p + "attn_qkv.weight",  4096*6144, GGML_TYPE_Q4_K, -1));
        m.tensors.push_back(make_tensor(p + "attn_output.weight", 4096*4096, GGML_TYPE_Q4_K, -1));
        if (moe) {
            m.tensors.push_back(make_tensor(p + "ffn_gate_inp.weight", 4096*128, GGML_TYPE_F32, -1));
            m.tensors.push_back(make_tensor(p + "ffn_up_exps.weight",   128ull*4096*768, GGML_TYPE_Q4_K, -1));
            m.tensors.push_back(make_tensor(p + "ffn_gate_exps.weight", 128ull*4096*768, GGML_TYPE_Q4_K, -1));
            m.tensors.push_back(make_tensor(p + "ffn_down_exps.weight", 128ull*4096*768, GGML_TYPE_Q4_K, -1));
        } else {
            m.tensors.push_back(make_tensor(p + "ffn_up.weight",   4096*14336, GGML_TYPE_Q4_K, -1));
            m.tensors.push_back(make_tensor(p + "ffn_gate.weight", 4096*14336, GGML_TYPE_Q4_K, -1));
            m.tensors.push_back(make_tensor(p + "ffn_down.weight", 4096*14336, GGML_TYPE_Q4_K, -1));
        }
    }
    m.tensors.push_back(make_tensor("output_norm.weight", 4096, GGML_TYPE_F32, -1));
    m.tensors.push_back(make_tensor("output.weight", 600000000, GGML_TYPE_Q4_K, -1));
    return m;
}

static void place(hp_model & m, int32_t dev, bool (*pred)(const hp_tensor &)) {
    for (hp_tensor & t : m.tensors) {
        if (pred(t)) {
            t.dev = dev;
        }
    }
}

static size_t gpu_bytes(const hp_model & m) {
    size_t n = 0;
    for (const hp_tensor & t : m.tensors) {
        n += t.dev >= 0 ? t.bytes : 0;
    }
    return n;
}

static void test_classify() {
    CHECK(hp_classify("blk.3.ffn_up_exps.weight")      == HP_USE_EXPERT);
    CHECK(hp_classify("blk.3.ffn_gate_up_exps.weight") == HP_USE_EXPERT);
    CHECK(hp_classify("blk.3.ffn_down_chexps.weight")  == HP_USE_EXPERT);
    CHECK(hp_classify("blk.3.ffn_up_shexp.weight")     == HP_USE_DENSE);
    CHECK(hp_classify("blk.3.ffn_up.weight")           == HP_USE_DENSE);
    CHECK(hp_classify("token_embd.weight")             == HP_USE_ROWS);
    CHECK(hp_classify("per_layer_token_embd.weight")   == HP_USE_ROWS);
    CHECK(hp_classify("output.weight")                 == HP_USE_DENSE);

    CHECK(hp_layer_of("blk.0.attn_q.weight")   == 0);
    CHECK(hp_layer_of("blk.47.attn_q.weight")  == 47);
    CHECK(hp_layer_of("output.weight")         == -1);
    CHECK(hp_layer_of("blk.x.attn_q.weight")   == -1);
    CHECK(hp_layer_of("blk.12")                == -1);
}

static void test_dense() {
    const common_hw_profile prof = mock_profile();
    hp_cost_params cp;

    hp_model cpu_only = make_model(false, 32);
    hp_model gpu_all  = make_model(false, 32);
    place(gpu_all, 0, [](const hp_tensor &) { return true; });

    const hp_estimate e_cpu = hp_estimate_cost(cpu_only, prof, cp);
    const hp_estimate e_gpu = hp_estimate_cost(gpu_all,  prof, cp);
    CHECK(e_cpu.warnings.empty());
    CHECK(e_gpu.tg_tps > 5.0*e_cpu.tg_tps);
    CHECK(e_gpu.pp_tps > e_cpu.pp_tps);
    CHECK(e_cpu.n_switch == 0);
    CHECK(e_gpu.n_switch == 0);
    printf("dense 8B: cpu %.1f t/s, gpu %.1f t/s decode, pp %.0f vs %.0f t/s\n", e_cpu.tg_tps, e_gpu.tg_tps, e_cpu.pp_tps, e_gpu.pp_tps);

    // the cost of a dense placement grows when any single tensor moves from the GPU to the host
    hp_model half = make_model(false, 32);
    place(half, 0, [](const hp_tensor & t) { return t.layer >= 16 || t.layer < 0; });
    const hp_estimate e_half = hp_estimate_cost(half, prof, cp);
    CHECK(e_half.tg_tps > e_cpu.tg_tps && e_half.tg_tps < e_gpu.tg_tps);
    for (size_t i = 0; i < half.tensors.size(); i++) {
        if (half.tensors[i].dev < 0 || half.tensors[i].bytes < 1024*1024) {
            continue;
        }
        hp_model moved = half;
        moved.tensors[i].dev = -1;
        CHECK(hp_estimate_cost(moved, prof, cp).tg_s > e_half.tg_s);
    }

    // alternating layers costs device switches that a contiguous split does not
    hp_model alt = make_model(false, 32);
    place(alt, 0, [](const hp_tensor & t) { return t.layer % 2 == 0 || t.layer < 0; });
    const hp_estimate e_alt = hp_estimate_cost(alt, prof, cp);
    CHECK(gpu_bytes(alt) > 0);
    CHECK(e_alt.n_switch > e_half.n_switch);
    CHECK(e_half.n_switch <= 2);
}

static void test_moe() {
    const common_hw_profile prof = mock_profile();
    hp_cost_params cp;

    // with the same VRAM, dense tensors on the GPU beat experts on the GPU: each expert byte is read 1/16 as often
    hp_model dense_first = make_model(true, 48);
    place(dense_first, 0, [](const hp_tensor & t) { return t.use == HP_USE_DENSE; });
    const size_t budget = gpu_bytes(dense_first);

    hp_model experts_first = make_model(true, 48);
    size_t used = 0;
    for (hp_tensor & t : experts_first.tensors) {
        if (t.use == HP_USE_EXPERT && used + t.bytes <= budget) {
            t.dev = 0;
            used += t.bytes;
        }
    }
    CHECK(used > budget/2);

    const hp_estimate e_dense   = hp_estimate_cost(dense_first,   prof, cp);
    const hp_estimate e_experts = hp_estimate_cost(experts_first, prof, cp);
    CHECK(e_dense.tg_tps > e_experts.tg_tps);
    printf("moe 30B-A3B: dense on gpu %.1f t/s, same vram as experts %.1f t/s\n", e_dense.tg_tps, e_experts.tg_tps);

    // prefill with host experts: a large ubatch copies them to the GPU, a small one computes on the CPU
    hp_cost_params cp_small = cp;
    cp_small.n_ubatch = 16;
    hp_cost_params cp_no_offload = cp;
    cp_no_offload.op_offload = false;
    const hp_estimate e_pp_small   = hp_estimate_cost(dense_first, prof, cp_small);
    const hp_estimate e_pp_offload = hp_estimate_cost(dense_first, prof, cp);
    const hp_estimate e_pp_cpu     = hp_estimate_cost(dense_first, prof, cp_no_offload);
    CHECK(e_pp_small.pp_tps > 0.0);
    CHECK(e_pp_offload.pp_tps > e_pp_cpu.pp_tps);
    printf("moe prefill at ub 512: offload %.0f t/s, cpu only %.0f t/s, ub 16 %.0f t/s\n",
        e_pp_offload.pp_tps, e_pp_cpu.pp_tps, e_pp_small.pp_tps);

    // decode does not depend on the ubatch
    CHECK(std::fabs(e_pp_small.tg_s - e_pp_offload.tg_s) < 1e-12);

    // a dense model reads every expert once per token if n_expert is unknown
    hp_model no_meta = dense_first;
    no_meta.n_expert = 0;
    CHECK(hp_estimate_cost(no_meta, prof, cp).tg_s > e_dense.tg_s);
}

static void test_context() {
    const common_hw_profile prof = mock_profile();
    hp_cost_params cp;

    hp_model m = make_model(false, 8);
    place(m, 0, [](const hp_tensor &) { return true; });
    const double base = hp_estimate_cost(m, prof, cp).tg_s;

    // a KV cache on the GPU costs less per token than the same cache in system memory
    hp_model kv_gpu = m;
    kv_gpu.ctx_bytes = {1ull << 30, 0};
    hp_model kv_host = m;
    kv_host.ctx_bytes = {0, 1ull << 30};
    const double t_gpu  = hp_estimate_cost(kv_gpu,  prof, cp).tg_s;
    const double t_host = hp_estimate_cost(kv_host, prof, cp).tg_s;
    CHECK(t_gpu > base);
    CHECK(t_host > t_gpu);
}

static void test_missing_device() {
    common_hw_profile prof = mock_profile();
    prof.devs.pop_back();
    hp_model m = make_model(false, 2);
    place(m, 0, [](const hp_tensor &) { return true; });
    const hp_estimate e = hp_estimate_cost(m, prof, hp_cost_params());
    CHECK(!e.warnings.empty());

    common_hw_profile empty;
    CHECK(!hp_estimate_cost(m, empty, hp_cost_params()).warnings.empty());
}

// load every generated test model with no_alloc and check that the collected tensors make sense
static void test_models(const std::string & dir) {
    common_hw_profile prof = mock_profile();
    int n_models = 0;
    int n_moe    = 0;
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() != ".gguf") {
            continue;
        }
        const std::string path = entry.path().string();

        llama_model_params   mp = llama_model_default_params();
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 256;

        hp_model m;
        if (!hp_model_from_file(path.c_str(), mp, cp, m)) {
            fprintf(stderr, "skip %s: failed to load\n", path.c_str());
            continue;
        }
        n_models++;

        size_t n_expert_tensors = 0;
        size_t n_layer_tensors  = 0;
        size_t total            = 0;
        for (const hp_tensor & t : m.tensors) {
            n_expert_tensors += t.use == HP_USE_EXPERT;
            n_layer_tensors  += t.layer >= 0;
            total            += t.bytes;
            CHECK(t.layer < (int32_t) m.n_layer + 8); // NextN/MTP blocks are appended after the trunk
            CHECK(t.dev == -1); // there is no GPU here
        }
        CHECK(!m.tensors.empty());
        CHECK(n_layer_tensors > 0);
        if (m.n_expert > 0 && n_expert_tensors > 0) {
            n_moe++;
            CHECK(m.n_expert_used > 0 && m.n_expert_used <= m.n_expert);
        }

        const hp_estimate e = hp_estimate_cost(m, prof, hp_cost_params());
        CHECK(e.bytes_dev.back() == total);
        CHECK(e.tg_s > 0.0 && std::isfinite(e.tg_s));
        CHECK(e.pp_s > 0.0 && std::isfinite(e.pp_s));
    }
    printf("estimated %d generated models, %d of them MoE\n", n_models, n_moe);
    CHECK(n_models > 0);
    CHECK(n_moe > 0);
}

int main(int argc, char ** argv) {
    std::string models_dir;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0 && i + 1 < argc) {
            models_dir = argv[++i];
        }
    }

    test_classify();
    test_dense();
    test_moe();
    test_context();
    test_missing_device();

    if (!models_dir.empty()) {
        llama_backend_init();
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
        test_models(models_dir);
        llama_backend_free();
    }

    if (n_fail > 0) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
