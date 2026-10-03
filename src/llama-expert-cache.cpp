#include "llama-expert-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

static std::vector<ggml_backend_buffer_type_t> llama_expert_cache_cpu_extra_bufts() {
    std::vector<ggml_backend_buffer_type_t> ret;
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        return ret;
    }
    auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto get_extra_bufts = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra_bufts) {
        for (ggml_backend_buffer_type_t * b = get_extra_bufts(cpu_dev); b && *b; ++b) {
            ret.push_back(*b);
        }
    }
    return ret;
}

llama_expert_cache::llama_expert_cache(const llama_model & model, int32_t n_slots, bool allow_host) : n_slots_(n_slots) {
    if (n_slots <= 0) {
        return;
    }

    const auto extra_bufts = llama_expert_cache_cpu_extra_bufts();

    // one context and buffer per device buffer type
    struct buft_layers {
        std::vector<int32_t> il;
    };
    std::map<ggml_backend_buffer_type_t, buft_layers> by_buft;

    const int32_t n_layer = (int32_t) model.layers.size();
    for (int32_t il = 0; il < n_layer; il++) {
        const llama_layer & layer = model.layers[il];
        ggml_tensor * src[llama_expert_cache_layer::N_PROJ] = {
            layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, layer.ffn_gate_up_exps,
        };

        bool any = false;
        bool ok  = true;
        int64_t n_expert = 0;
        for (ggml_tensor * t : src) {
            if (t == nullptr) {
                continue;
            }
            any = true;
            // only stacks in plain system memory: a device already holds the others, and a repacked layout cannot be copied by expert
            if (t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer) || t->data == nullptr ||
                std::find(extra_bufts.begin(), extra_bufts.end(), ggml_backend_buffer_get_type(t->buffer)) != extra_bufts.end() ||
                !ggml_is_contiguous(t) || ggml_n_dims(t) != 3) {
                ok = false;
            }
            n_expert = n_expert == 0 ? t->ne[2] : n_expert;
            ok = ok && t->ne[2] == n_expert;
        }
        if (!any || !ok || n_expert <= n_slots) {
            continue;
        }

        ggml_backend_dev_t dev = model.dev_layer(il);
        if (dev == nullptr) {
            continue;
        }
        const bool dev_is_host = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (dev_is_host && !allow_host) {
            continue;
        }
        by_buft[ggml_backend_dev_buffer_type(dev)].il.push_back(il);

        llama_expert_cache_layer & l = layers[il];
        l.il = il;
        for (int p = 0; p < llama_expert_cache_layer::N_PROJ; p++) {
            l.src[p] = src[p];
        }
        l.slot_expert.assign(n_slots, -1);
        l.expert_slot.assign(n_expert, -1);
        l.score.assign(n_expert, 0.0f);
    }

    for (auto & [buft, bl] : by_buft) {
        ggml_init_params params = {
            /*.mem_size   =*/ bl.il.size()*(llama_expert_cache_layer::N_PROJ + 4)*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create the expert cache context");
        }
        ctxs.emplace_back(ctx);

        for (int32_t il : bl.il) {
            llama_expert_cache_layer & l = layers[il];
            const int64_t n_expert = (int64_t) l.expert_slot.size();
            for (int p = 0; p < llama_expert_cache_layer::N_PROJ; p++) {
                if (l.src[p] == nullptr) {
                    continue;
                }
                l.dst[p] = ggml_new_tensor_3d(ctx, l.src[p]->type, l.src[p]->ne[0], l.src[p]->ne[1], n_slots);
                ggml_format_name(l.dst[p], "%s.cache", ggml_get_name(l.src[p]));
            }
            l.tab_slot = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_expert);
            l.tab_host = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_expert);
            l.tab_hit  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_expert);
            l.tab_miss = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_expert);
            ggml_format_name(l.tab_slot, "blk.%d.expert_cache_slot", il);
            ggml_format_name(l.tab_host, "blk.%d.expert_cache_host", il);
            ggml_format_name(l.tab_hit,  "blk.%d.expert_cache_hit",  il);
            ggml_format_name(l.tab_miss, "blk.%d.expert_cache_miss", il);
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buf == nullptr) {
            throw std::runtime_error("failed to allocate the expert cache buffer");
        }
        // zeros are valid data of every type, slot 0 is read for misses before anything is uploaded
        ggml_backend_buffer_clear(buf, 0);
        stats_.n_bytes += ggml_backend_buffer_get_size(buf);
        bufs.emplace_back(buf);

        for (int32_t il : bl.il) {
            upload_tables(layers[il]);
        }
    }

    stats_.n_layers = (int32_t) layers.size();
    if (enabled()) {
        LLAMA_LOG_INFO("%s: caching %d experts per layer for %d layers, %.2f MiB\n",
            __func__, n_slots, stats_.n_layers, stats_.n_bytes/(1024.0*1024.0));
    } else {
        LLAMA_LOG_WARN("%s: no layer qualifies: the expert stacks must be in plain system memory (not repacked) next to a GPU layer\n", __func__);
    }
}

const llama_expert_cache_layer * llama_expert_cache::get(int32_t il) const {
    auto it = layers.find(il);
    return it == layers.end() ? nullptr : &it->second;
}

void llama_expert_cache::observe(int32_t il, const int32_t * ids, int64_t n) {
    auto it = layers.find(il);
    if (it == layers.end()) {
        return;
    }
    llama_expert_cache_layer & l = it->second;
    const int32_t n_expert = (int32_t) l.score.size();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert) {
            continue;
        }
        l.score[e] += 1.0f;
        stats_.n_lookups++;
        stats_.n_hits += l.expert_slot[e] >= 0;
    }
}

void llama_expert_cache::upload_tables(llama_expert_cache_layer & l) {
    const int32_t n_expert = (int32_t) l.expert_slot.size();

    // the stand-in for hits on the host is the most used expert that is not cached: it is the one most likely read anyway
    int32_t stand_in = 0;
    float   best     = -1.0f;
    for (int32_t e = 0; e < n_expert; e++) {
        if (l.expert_slot[e] < 0 && l.score[e] > best) {
            best     = l.score[e];
            stand_in = e;
        }
    }

    std::vector<int32_t> slot(n_expert);
    std::vector<int32_t> host(n_expert);
    std::vector<float>   hit(n_expert);
    std::vector<float>   miss(n_expert);
    for (int32_t e = 0; e < n_expert; e++) {
        const bool cached = l.expert_slot[e] >= 0;
        slot[e] = cached ? l.expert_slot[e] : 0;
        host[e] = cached ? stand_in : e;
        hit[e]  = cached ? 1.0f : 0.0f;
        miss[e] = cached ? 0.0f : 1.0f;
    }
    ggml_backend_tensor_set(l.tab_slot, slot.data(), 0, ggml_nbytes(l.tab_slot));
    ggml_backend_tensor_set(l.tab_host, host.data(), 0, ggml_nbytes(l.tab_host));
    ggml_backend_tensor_set(l.tab_hit,  hit.data(),  0, ggml_nbytes(l.tab_hit));
    ggml_backend_tensor_set(l.tab_miss, miss.data(), 0, ggml_nbytes(l.tab_miss));
}

void llama_expert_cache::update(bool force) {
    if (!enabled()) {
        return;
    }
    if (!force && ++n_calls < interval) {
        return;
    }
    n_calls = 0;

    int32_t budget = max_uploads;
    for (auto & [il, l] : layers) {
        const int32_t n_expert = (int32_t) l.score.size();

        std::vector<int32_t> order(n_expert);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return l.score[a] > l.score[b]; });

        bool changed = false;
        for (int32_t k = 0; k < n_slots_ && budget > 0; k++) {
            const int32_t e = order[k];
            if (l.score[e] <= 0.0f || l.expert_slot[e] >= 0) {
                continue;
            }
            // an empty slot, else the cached expert with the lowest score
            int32_t victim = -1;
            for (int32_t s = 0; s < n_slots_; s++) {
                if (l.slot_expert[s] < 0) {
                    victim = s;
                    break;
                }
                if (victim < 0 || l.score[l.slot_expert[s]] < l.score[l.slot_expert[victim]]) {
                    victim = s;
                }
            }
            const int32_t old = l.slot_expert[victim];
            // replace only for a clear gain, so that two close experts do not swap back and forth
            if (old >= 0 && l.score[e] <= 1.25f*l.score[old]) {
                continue;
            }
            for (int p = 0; p < llama_expert_cache_layer::N_PROJ; p++) {
                if (l.src[p] == nullptr) {
                    continue;
                }
                const size_t sz = l.src[p]->nb[2];
                ggml_backend_tensor_set(l.dst[p], (const char *) l.src[p]->data + e*sz, victim*sz, sz);
            }
            if (old >= 0) {
                l.expert_slot[old] = -1;
            }
            l.slot_expert[victim] = e;
            l.expert_slot[e]      = victim;
            stats_.n_uploads++;
            budget--;
            changed = true;
        }

        // the stand-in follows the scores even when no expert moved
        GGML_UNUSED(changed);
        upload_tables(l);

        // forget slowly, so that the cache follows a change of topic
        for (float & s : l.score) {
            s *= 0.5f;
        }
    }
}
