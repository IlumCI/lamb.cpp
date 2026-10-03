// Tests for the MoE expert cache (n_expert_cache).
// Every generated MoE test model is run twice on the CPU, without and with a cache that is placed in host memory
// (LLAMA_EXPERT_CACHE_HOST=1) and updated after every ubatch (LLAMA_EXPERT_CACHE_INTERVAL=1). The cached path computes
// hits from the copies and misses from the original stacks, and selects with x*1 + y*0, so the logits must be bit-identical.

#include "llama.h"
#include "../src/llama-ext.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

// prompt in one batch, then single tokens; returns the logits of every step
static std::vector<std::vector<float>> run(llama_model * model, int32_t n_expert_cache, const std::vector<llama_token> & tokens,
        int32_t n_prompt, llama_expert_cache_info * info) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 256;
    cp.n_batch         = 64;
    cp.n_ubatch        = 64;
    cp.n_threads       = 2;
    cp.n_threads_batch = 2;
    cp.n_expert_cache  = n_expert_cache;

    std::vector<std::vector<float>> ret;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        return ret;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> prompt(tokens.begin(), tokens.begin() + n_prompt);
    if (llama_decode(ctx, llama_batch_get_one(prompt.data(), (int32_t) prompt.size())) != 0) {
        llama_free(ctx);
        return ret;
    }
    ret.emplace_back(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);

    for (size_t i = n_prompt; i < tokens.size(); i++) {
        llama_token t = tokens[i];
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) {
            break;
        }
        ret.emplace_back(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);
    }

    if (info) {
        *info = {};
        llama_expert_cache_get_info(ctx, info);
    }
    llama_free(ctx);
    return ret;
}

static bool same(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
    if (a.size() != b.size() || a.empty()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].size() != b[i].size() || memcmp(a[i].data(), b[i].data(), a[i].size()*sizeof(float)) != 0) {
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

    set_env("LLAMA_EXPERT_CACHE_HOST", "1");
    set_env("LLAMA_EXPERT_CACHE_INTERVAL", "1");

    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    int n_models = 0;
    int n_cached = 0;
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        const std::string path = entry.path().string();
        if (path.size() < 9 || path.substr(path.size() - 9) != "-moe.gguf") {
            continue;
        }

        llama_model_params mp = llama_model_default_params();
        llama_model * model = llama_model_load_from_file(path.c_str(), mp);
        if (model == nullptr) {
            continue;
        }
        const int32_t n_expert = llama_model_n_expert(model);
        const int32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
        if (n_expert < 2 || n_vocab < 16) {
            fprintf(stderr, "skip %s: n_expert %d, n_vocab %d\n", path.c_str(), n_expert, n_vocab);
            llama_model_free(model);
            continue;
        }

        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> tok(1, n_vocab - 1);
        std::vector<llama_token> tokens(40);
        for (auto & t : tokens) {
            t = tok(rng);
        }

        const auto ref = run(model, 0, tokens, 8, nullptr);
        if (ref.empty()) {
            // the model cannot run a plain decode in this setting, nothing to compare
            fprintf(stderr, "skip %s: decode failed\n", path.c_str());
            llama_model_free(model);
            continue;
        }
        n_models++;

        llama_expert_cache_info info = {};
        const auto got = run(model, n_expert/2 > 0 ? n_expert/2 : 1, tokens, 8, &info);
        const bool ok = same(ref, got);
        CHECK(ok);
        // a model may load expert stacks that its graph does not route through build_moe_ffn
        if (info.n_layers > 0 && info.n_lookups > 0) {
            n_cached++;
            CHECK(info.n_uploads > 0);
            CHECK(info.n_hits > 0);
        }
        printf("%-40s %s  layers %d, lookups %lld, hits %lld, uploads %lld\n",
            entry.path().filename().string().c_str(), ok ? "same" : "DIFFERENT",
            info.n_layers, (long long) info.n_lookups, (long long) info.n_hits, (long long) info.n_uploads);

        llama_model_free(model);
    }

    printf("%d MoE models compared, %d of them used the cache\n", n_models, n_cached);
    CHECK(n_models > 0);
    CHECK(n_cached > n_models/2);

    llama_backend_free();

    if (n_fail > 0) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
