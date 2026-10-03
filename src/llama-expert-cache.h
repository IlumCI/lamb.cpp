#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <map>
#include <vector>

struct llama_model;

// A cache of routed MoE experts in device memory, for layers whose expert stacks live in system memory.
// The graph computes each used expert either from the cache (hit) or from system memory (miss):
//   - tab_slot maps an expert to its cache slot, a miss maps to slot 0 and its result is discarded
//   - tab_host maps a miss to itself and a hit to one stand-in expert, so the host reads only the missed experts
//   - tab_hit / tab_miss are 1 and 0 per expert and select the result
// Between graphs, update() moves the most used experts into the slots.
struct llama_expert_cache_layer {
    int32_t il = -1;

    // up, gate, down, gate_up: the expert stacks in system memory and their cache in device memory
    static constexpr int N_PROJ = 4;
    ggml_tensor * src[N_PROJ] = {};
    ggml_tensor * dst[N_PROJ] = {};

    ggml_tensor * tab_slot = nullptr; // I32 [1, n_expert]
    ggml_tensor * tab_host = nullptr; // I32 [1, n_expert]
    ggml_tensor * tab_hit  = nullptr; // F32 [1, n_expert]
    ggml_tensor * tab_miss = nullptr; // F32 [1, n_expert]

    std::vector<int32_t> slot_expert; // expert in each slot, -1 if empty
    std::vector<int32_t> expert_slot; // slot of each expert, -1 if not cached
    std::vector<float>   score;       // decayed use count of each expert
};

struct llama_expert_cache_stats {
    int64_t n_lookups = 0; // routed expert uses seen
    int64_t n_hits    = 0; // of those, served from the cache
    int64_t n_uploads = 0; // experts copied into a slot
    size_t  n_bytes   = 0; // device memory of the cache
    int32_t n_layers  = 0; // layers with a cache
};

class llama_expert_cache {
public:
    // n_slots experts per layer; allow_host also caches layers whose device is the CPU (for tests)
    llama_expert_cache(const llama_model & model, int32_t n_slots, bool allow_host);

    bool    enabled() const { return !layers.empty(); }
    int32_t n_slots() const { return n_slots_; }

    // cache of a layer, nullptr if the layer has none
    const llama_expert_cache_layer * get(int32_t il) const;

    // count the experts a ubatch used in a layer, ids has n values
    void observe(int32_t il, const int32_t * ids, int64_t n);

    // call after each ubatch, every interval calls it moves hot experts into the cache and uploads the tables
    // no graph may run during this
    void update(bool force = false);

    // number of update() calls between real updates
    void set_interval(int32_t n) { interval = n > 0 ? n : 1; }

    const llama_expert_cache_stats & stats() const { return stats_; }

private:
    void upload_tables(llama_expert_cache_layer & l);

    int32_t n_slots_;
    int32_t interval     = 8;  // update() calls between real updates
    int32_t n_calls      = 0;
    int32_t max_uploads  = 64; // experts per update, bounds the stall

    std::map<int32_t, llama_expert_cache_layer> layers;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    llama_expert_cache_stats stats_;
};
