// Tests for the LoRA cache (llama_model_set_lora_cache / --lora-cache).
// Three random adapters for a generated test model are loaded into two copies of the model: one plain, and one with a cache
// whose budget fits two adapters, placed in host memory (LLAMA_LORA_CACHE_HOST=1). Switching between the adapters must
// give bit-identical logits whether an adapter is read from its home copy or its cached copy, and must upload and evict.

#include "llama.h"
#include "../src/llama-ext.h"

#include "ggml.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

// write a rank-4 LoRA for every attn_q and ffn_up weight of the base model, returns its tensor bytes
static size_t write_lora(const std::string & base, const std::string & path, int seed) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * src = gguf_init_from_file(base.c_str(), gp);
    if (src == nullptr) {
        return 0;
    }
    const std::string arch = gguf_get_val_str(src, gguf_find_key(src, "general.architecture"));

    const int64_t rank = 4;
    std::vector<std::pair<std::string, ggml_tensor *>> targets;
    for (ggml_tensor * t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
        const std::string name = ggml_get_name(t);
        if (name.find("attn_q.weight") != std::string::npos || name.find("ffn_up.weight") != std::string::npos) {
            targets.emplace_back(name, t);
        }
    }

    ggml_init_params ip = { 2*targets.size()*ggml_tensor_overhead() + 64ull*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * dst = gguf_init_empty();
    gguf_set_val_str(dst, "general.type", "adapter");
    gguf_set_val_str(dst, "general.architecture", arch.c_str());
    gguf_set_val_str(dst, "adapter.type", "lora");
    gguf_set_val_f32(dst, "adapter.lora.alpha", 8.0f);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    size_t bytes = 0;
    for (const auto & [name, t] : targets) {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t->ne[0], rank);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, t->ne[1]);
        ggml_set_name(a, (name + ".lora_a").c_str());
        ggml_set_name(b, (name + ".lora_b").c_str());
        for (ggml_tensor * x : {a, b}) {
            float * d = (float *) x->data;
            for (int64_t i = 0; i < ggml_nelements(x); i++) {
                d[i] = dist(rng);
            }
            gguf_add_tensor(dst, x);
            bytes += ggml_nbytes(x);
        }
    }
    gguf_write_to_file(dst, path.c_str(), false);

    gguf_free(dst);
    ggml_free(ctx);
    gguf_free(src);
    ggml_free(meta);
    return targets.empty() ? 0 : bytes;
}

static std::vector<std::vector<float>> run(llama_context * ctx, llama_model * model, const std::vector<llama_token> & tokens) {
    llama_memory_clear(llama_get_memory(ctx), true);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<std::vector<float>> ret;
    std::vector<llama_token> prompt(tokens.begin(), tokens.begin() + 8);
    if (llama_decode(ctx, llama_batch_get_one(prompt.data(), (int32_t) prompt.size())) != 0) {
        return ret;
    }
    ret.emplace_back(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);
    for (size_t i = 8; i < tokens.size(); i++) {
        llama_token t = tokens[i];
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) {
            break;
        }
        ret.emplace_back(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);
    }
    return ret;
}

static bool same(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
    if (a.size() != b.size() || a.empty()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (memcmp(a[i].data(), b[i].data(), a[i].size()*sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string dir;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0 && i + 1 < argc) {
            dir = argv[++i];
        }
    }
    if (dir.empty()) {
        fprintf(stderr, "usage: %s --models DIR\n", argv[0]);
        return 1;
    }
    const std::string base = dir + "/llama-dense.gguf";

#ifdef _WIN32
    _putenv_s("LLAMA_LORA_CACHE_HOST", "1");
#else
    setenv("LLAMA_LORA_CACHE_HOST", "1", 1);
#endif

    const auto tmp = std::filesystem::temp_directory_path() / "test-lora-cache";
    std::filesystem::create_directories(tmp);
    std::vector<std::string> paths;
    size_t lora_bytes = 0;
    for (int k = 0; k < 3; k++) {
        paths.push_back((tmp / ("lora-" + std::to_string(k) + ".gguf")).string());
        lora_bytes = write_lora(base, paths.back(), 10 + k);
    }
    CHECK(lora_bytes > 0);

    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    llama_model_params mp = llama_model_default_params();
    llama_model * model_ref = llama_model_load_from_file(base.c_str(), mp);
    llama_model * model_c   = llama_model_load_from_file(base.c_str(), mp);
    CHECK(model_ref && model_c);
    if (!model_ref || !model_c) {
        return 1;
    }

    // room for two adapters and some padding, not for three
    const size_t budget = lora_bytes*2 + lora_bytes/2;
    llama_model_set_lora_cache(model_c, budget);

    std::vector<llama_adapter_lora *> ref, cached;
    for (const auto & p : paths) {
        ref.push_back(llama_adapter_lora_init(model_ref, p.c_str()));
        cached.push_back(llama_adapter_lora_init(model_c, p.c_str()));
        CHECK(ref.back() && cached.back());
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_threads = 2;
    cp.n_threads_batch = 2;
    llama_context * ctx_ref = llama_init_from_model(model_ref, cp);
    llama_context * ctx_c   = llama_init_from_model(model_c, cp);

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_ref));
    std::mt19937 rng(7);
    std::uniform_int_distribution<int32_t> tok(1, n_vocab - 1);
    std::vector<llama_token> tokens(20);
    for (auto & t : tokens) {
        t = tok(rng);
    }

    // without an adapter both models agree; with one the logits change, so the comparison below means something
    const auto base_out = run(ctx_ref, model_ref, tokens);
    CHECK(same(base_out, run(ctx_c, model_c, tokens)));

    const int order[] = {0, 1, 2, 0, 1, 2, 2, 0};
    for (int step = 0; step < (int) (sizeof(order)/sizeof(order[0])); step++) {
        const int k = order[step];
        float scale = 1.0f;
        llama_set_adapters_lora(ctx_ref, &ref[k], 1, &scale);
        llama_set_adapters_lora(ctx_c, &cached[k], 1, &scale);

        // odd steps decode at once, while the upload may be in flight; even steps give the worker time first
        if (step % 2 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        const auto r = run(ctx_ref, model_ref, tokens);
        const auto c = run(ctx_c, model_c, tokens);
        CHECK(!same(r, base_out));
        CHECK(same(r, c));

        llama_lora_cache_info info = {};
        CHECK(llama_model_get_lora_cache_info(model_c, &info));
        CHECK(info.n_bytes <= budget);
        printf("step %d adapter %d: %s, uploads %lld, evictions %lld, resident %d, %zu of %zu bytes\n", step, k,
            same(r, c) ? "same" : "DIFFERENT", (long long) info.n_uploads, (long long) info.n_evictions, info.n_resident, info.n_bytes, budget);
    }

    llama_lora_cache_info info = {};
    llama_model_get_lora_cache_info(model_c, &info);
    CHECK(info.n_uploads >= 3);
    CHECK(info.n_evictions >= 1);
    CHECK(info.n_failed == 0);
    CHECK(info.n_resident <= 2);

    // freeing an adapter while it is cached must not leave the cache pointing at it
    llama_adapter_lora_free(cached[0]);
    llama_model_get_lora_cache_info(model_c, &info);
    CHECK(info.n_bytes <= budget);

    llama_free(ctx_c);
    llama_free(ctx_ref);
    llama_model_free(model_c);
    llama_model_free(model_ref);
    llama_backend_free();

    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);

    if (n_fail > 0) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
