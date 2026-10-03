#pragma once

#include "hw-profile.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

// Throughput cost model for a placement of model weights across devices and system memory.
// It predicts tokens/s for decode and prefill from the hardware profile, so that placements can be compared.
// Only weight matmuls and the read of the context memory are counted, attention compute is not.

enum hp_tensor_use {
    HP_USE_DENSE,  // read in full for every token
    HP_USE_EXPERT, // routed MoE expert stack, only n_expert_used of n_expert are read per token
    HP_USE_ROWS,   // embedding table, only the rows of the input tokens are read
};

struct hp_tensor {
    std::string   name;
    int32_t       layer    = -1; // -1 for tensors outside the repeating blocks
    enum ggml_type type    = GGML_TYPE_F32;
    int64_t       n_params = 0;
    int64_t       n_rows   = 1;  // rows of an embedding table
    size_t        bytes    = 0;
    hp_tensor_use use      = HP_USE_DENSE;
    int32_t       dev      = -1; // index into hp_model::dev_names, -1 for system memory
};

struct hp_model {
    int32_t n_layer       = 0;
    int32_t n_expert      = 0;
    int32_t n_expert_used = 0;

    std::vector<std::string> dev_names;  // model devices, in model order
    std::vector<size_t>      ctx_bytes;  // context memory (KV cache, recurrent state) per device, last entry is system memory
    std::vector<hp_tensor>   tensors;    // in model order
};

struct hp_cost_params {
    int32_t n_ubatch             = 512;
    double  ctx_fill             = 0.5;   // fraction of the context memory read per decoded token
    double  t_switch_s           = 20e-6; // cost of each change of device during a forward pass
    bool    op_offload           = true;  // large batches copy host weights to the GPU, as ggml-backend does
    int32_t op_offload_min_batch = 32;    // GGML_OP_OFFLOAD_MIN_BATCH
};

struct hp_estimate {
    double  tg_s     = 0.0; // seconds per decoded token
    double  tg_tps   = 0.0;
    double  pp_s     = 0.0; // seconds per ubatch of prompt
    double  pp_tps   = 0.0;
    int32_t n_switch = 0;   // device changes per forward pass

    std::vector<double> tg_s_dev;  // decode seconds per device, last entry is system memory
    std::vector<size_t> bytes_dev; // weight bytes per device, last entry is system memory

    std::vector<std::string> warnings;
};

hp_tensor_use hp_classify(const std::string & name);
int32_t       hp_layer_of(const std::string & name);

// collect the tensors and their placement from a model loaded with no_alloc
void hp_model_from_llama(const llama_model * model, const llama_context * ctx, hp_model & out);

// load path with no_alloc and the given params, then collect, returns false on failure
bool hp_model_from_file(const char * path, const llama_model_params & mparams, const llama_context_params & cparams, hp_model & out);

hp_estimate hp_estimate_cost(const hp_model & model, const common_hw_profile & prof, const hp_cost_params & params);

std::string hp_estimate_to_string(const hp_model & model, const hp_estimate & est);

// estimate the placement given by mparams/cparams and log it
//   - profile_path: profile JSON to use, empty to use the cached profile of this machine (measured once if missing)
bool hp_log_estimate(const char * path_model, const llama_model_params & mparams, const llama_context_params & cparams,
        const std::string & profile_path, int32_t n_threads);
