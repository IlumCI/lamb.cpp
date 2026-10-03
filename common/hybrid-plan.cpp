#include "hybrid-plan.h"

#include "log.h"

#include "../src/llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

hp_tensor_use hp_classify(const std::string & name) {
    // routed expert stacks are named ffn_{up,gate,down,gate_up}_exps, chunked ones _chexps
    // shared experts (_shexp) are read in full and stay dense
    if (name.find("_exps.") != std::string::npos || name.find("_chexps.") != std::string::npos) {
        return HP_USE_EXPERT;
    }
    if (name.find("token_embd") != std::string::npos) {
        return HP_USE_ROWS;
    }
    return HP_USE_DENSE;
}

int32_t hp_layer_of(const std::string & name) {
    if (name.rfind("blk.", 0) != 0) {
        return -1;
    }
    const char * p   = name.c_str() + 4;
    char *       end = nullptr;
    const long   il  = strtol(p, &end, 10);
    if (end == p || *end != '.') {
        return -1;
    }
    return (int32_t) il;
}

void hp_model_from_llama(const llama_model * model, const llama_context * ctx, hp_model & out) {
    out = hp_model();
    out.n_layer       = llama_model_n_layer(model);
    out.n_expert      = llama_model_n_expert(model);
    out.n_expert_used = llama_model_n_expert_used(model);

    const int32_t nd = llama_model_n_devices(model);
    for (int32_t i = 0; i < nd; i++) {
        out.dev_names.push_back(ggml_backend_dev_name(llama_model_get_device(model, i)));
    }

    auto dev_index = [&](ggml_backend_buffer_type_t buft) -> int32_t {
        if (buft == nullptr || ggml_backend_buft_is_host(buft)) {
            return -1;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        for (int32_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model, i)) {
                return i;
            }
        }
        return -1;
    };

    const int32_t nt = llama_model_n_tensors(model);
    for (int32_t i = 0; i < nt; i++) {
        const ggml_tensor * t = llama_model_tensor_get(model, i);
        if (t == nullptr) {
            continue;
        }
        hp_tensor ht;
        ht.name     = ggml_get_name(t);
        ht.layer    = hp_layer_of(ht.name);
        ht.type     = t->type;
        ht.n_params = ggml_nelements(t);
        ht.n_rows   = std::max<int64_t>(1, ggml_nrows(t));
        ht.bytes    = ggml_nbytes(t);
        ht.use      = hp_classify(ht.name);
        ht.dev      = t->buffer ? dev_index(ggml_backend_buffer_get_type(t->buffer)) : -1;
        out.tensors.push_back(std::move(ht));
    }

    out.ctx_bytes.assign(nd + 1, 0);
    if (ctx != nullptr) {
        for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
            const int32_t id = dev_index(buft);
            out.ctx_bytes[id < 0 ? nd : id] += mb.context;
        }
    }
}

bool hp_model_from_file(const char * path, const llama_model_params & mparams, const llama_context_params & cparams, hp_model & out) {
    llama_model_params mp = mparams;
    mp.no_alloc  = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        return false;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    hp_model_from_llama(model, ctx, out);
    if (ctx) {
        llama_free(ctx);
    }
    llama_model_free(model);
    return true;
}

hp_estimate hp_estimate_cost(const hp_model & model, const common_hw_profile & prof, const hp_cost_params & params) {
    hp_estimate est;

    const size_t nd = model.dev_names.size();
    est.tg_s_dev.assign(nd + 1, 0.0);
    est.bytes_dev.assign(nd + 1, 0);

    const common_hw_dev_profile * cpu = prof.cpu();
    if (cpu == nullptr) {
        est.warnings.push_back("profile has no CPU entry");
        return est;
    }

    std::vector<const common_hw_dev_profile *> dps(nd, nullptr);
    for (size_t i = 0; i < nd; i++) {
        dps[i] = prof.find(model.dev_names[i]);
        if (dps[i] == nullptr) {
            est.warnings.push_back("profile has no entry for device " + model.dev_names[i]);
        }
    }

    // the GPU that large batches of host weights are copied to, the first one as in ggml-backend
    const common_hw_dev_profile * offload_dev = nullptr;
    for (size_t i = 0; i < nd && offload_dev == nullptr; i++) {
        if (dps[i] && !dps[i]->is_host() && dps[i]->h2d_gbps > 0.0) {
            offload_dev = dps[i];
        }
    }

    const double B        = std::max<int32_t>(1, params.n_ubatch);
    const double f_expert = model.n_expert > 0 && model.n_expert_used > 0 ?
        std::min(1.0, double(model.n_expert_used) / model.n_expert) : 1.0;
    // fraction of experts that at least one token of the ubatch uses, i.e. the expert weights a batch reads
    const double f_expert_pp = 1.0 - std::pow(1.0 - f_expert, B);

    const bool offload = params.op_offload && offload_dev != nullptr && params.n_ubatch >= params.op_offload_min_batch;

    ggml_backend_offload_params op = {};
    const bool use_cost = params.op_offload && params.offload_cost && offload_dev != nullptr &&
        common_hw_profile_offload_params(prof, offload_dev->name, op);

    int32_t prev_dev = INT32_MIN;
    for (const hp_tensor & t : model.tensors) {
        const size_t slot = t.dev < 0 ? nd : (size_t) t.dev;
        est.bytes_dev[slot] += t.bytes;

        const common_hw_dev_profile * dp = t.dev < 0 ? cpu : dps[t.dev];
        if (dp == nullptr) {
            continue;
        }

        // decode: one token reads every dense weight, a row of the embedding and the used experts
        double read = 0.0;
        switch (t.use) {
            case HP_USE_DENSE:  read = t.bytes;              break;
            case HP_USE_EXPERT: read = t.bytes*f_expert;     break;
            case HP_USE_ROWS:   read = double(t.bytes)/t.n_rows; break;
        }
        const double bw = dp->gemv_gbps_for(t.type)*1e9;
        if (bw > 0.0) {
            est.tg_s_dev[slot] += read/bw;
        }

        // count device changes along the weights of the forward pass, small tensors (norms, biases) follow their layer
        if (t.use != HP_USE_ROWS && t.bytes >= 64*1024) {
            if (prev_dev != INT32_MIN && t.dev != prev_dev) {
                est.n_switch++;
            }
            prev_dev = t.dev;
        }

        // prefill: flops of one ubatch, run on the device that holds the weight or copied to the GPU
        if (t.use == HP_USE_ROWS) {
            continue;
        }
        const double flops = 2.0*t.n_params*B*(t.use == HP_USE_EXPERT ? f_expert : 1.0);
        const double touch = t.bytes*(t.use == HP_USE_EXPERT ? f_expert_pp : 1.0);
        double t_host = 0.0;
        double t_dev  = 0.0;
        if (t.dev < 0 && use_cost && t.bytes > 0 && ggml_backend_offload_cost(&op, t.type, t.bytes, touch/t.bytes, flops, &t_host, &t_dev)) {
            est.pp_s += std::min(t_host, t_dev);
        } else if (t.dev < 0 && offload) {
            const double tflops = offload_dev->gemm_tflops_for(t.type)*1e12;
            const double gemv   = offload_dev->gemv_gbps_for(t.type)*1e9;
            const double t_mm   = tflops > 0.0 ? flops/tflops : 0.0;
            const double t_rd   = gemv   > 0.0 ? touch/gemv   : 0.0;
            est.pp_s += touch/(offload_dev->h2d_gbps*1e9) + std::max(t_mm, t_rd);
        } else {
            const double tflops = dp->gemm_tflops_for(t.type)*1e12;
            const double t_mm   = tflops > 0.0 ? flops/tflops : 0.0;
            const double t_rd   = bw > 0.0 ? touch/bw : 0.0;
            est.pp_s += std::max(t_mm, t_rd);
        }
    }

    // the context memory is read once per token, at the speed of the f16 matmul of the device that holds it
    for (size_t i = 0; i <= nd; i++) {
        const common_hw_dev_profile * dp = i == nd ? cpu : dps[i];
        if (dp == nullptr || model.ctx_bytes.size() <= i) {
            continue;
        }
        const double bw = dp->gemv_gbps_for(GGML_TYPE_F16)*1e9;
        if (bw > 0.0) {
            est.tg_s_dev[i] += model.ctx_bytes[i]*params.ctx_fill/bw;
        }
    }

    for (double s : est.tg_s_dev) {
        est.tg_s += s;
    }
    est.tg_s += est.n_switch*params.t_switch_s;
    est.pp_s += est.n_switch*params.t_switch_s;

    est.tg_tps = est.tg_s > 0.0 ? 1.0/est.tg_s : 0.0;
    est.pp_tps = est.pp_s > 0.0 ? B/est.pp_s   : 0.0;
    return est;
}

std::string hp_estimate_to_string(const hp_model & model, const hp_estimate & est) {
    constexpr double MiB = 1024.0*1024.0;
    std::string s;
    char buf[256];
    const size_t nd = model.dev_names.size();
    for (size_t i = 0; i <= nd; i++) {
        snprintf(buf, sizeof(buf), "  %-12s weights %9.1f MiB, context %8.1f MiB, decode %7.2f ms/token\n",
            i == nd ? "host" : model.dev_names[i].c_str(), est.bytes_dev[i]/MiB,
            model.ctx_bytes.size() > i ? model.ctx_bytes[i]/MiB : 0.0, est.tg_s_dev[i]*1e3);
        s += buf;
    }
    snprintf(buf, sizeof(buf), "  device switches per pass: %d\n", est.n_switch);
    s += buf;
    snprintf(buf, sizeof(buf), "  estimate: decode %.2f t/s, prefill %.1f t/s\n", est.tg_tps, est.pp_tps);
    s += buf;
    for (const std::string & w : est.warnings) {
        s += "  warning: " + w + "\n";
    }
    return s;
}

bool hp_log_estimate(const char * path_model, const llama_model_params & mparams, const llama_context_params & cparams,
        const std::string & profile_path, int32_t n_threads, bool offload_cost) {
    common_hw_profile prof;
    if (!common_hw_profile_resolve(profile_path, n_threads, prof)) {
        LOG_ERR("%s: failed to get a hardware profile\n", __func__);
        return false;
    }

    hp_model model;
    if (!hp_model_from_file(path_model, mparams, cparams, model)) {
        LOG_ERR("%s: failed to load model %s\n", __func__, path_model);
        return false;
    }

    hp_cost_params cp;
    cp.n_ubatch = cparams.n_ubatch;
    cp.op_offload = cparams.op_offload;
    cp.offload_cost = offload_cost;
    if (const char * env = getenv("GGML_OP_OFFLOAD_MIN_BATCH")) {
        cp.op_offload_min_batch = atoi(env);
    }

    const hp_estimate est = hp_estimate_cost(model, prof, cp);
    LOG_INF("%s: throughput estimate for this placement (weights and context reads only, n_ubatch = %d):\n%s",
        __func__, cp.n_ubatch, hp_estimate_to_string(model, est).c_str());
    return true;
}
