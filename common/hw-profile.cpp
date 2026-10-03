#include "hw-profile.h"

#include "common.h"
#include "json.h"
#include "log.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

//
// lookup helpers
//

bool common_hw_dev_profile::is_host() const {
    return type == GGML_BACKEND_DEVICE_TYPE_CPU || type == GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static ggml_type hw_type_by_name(const std::string & name) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * tn = ggml_type_name((ggml_type) t);
        if (tn != nullptr && name == tn) {
            return (ggml_type) t;
        }
    }
    return GGML_TYPE_COUNT;
}

static double hw_bits_per_weight(ggml_type type) {
    return 8.0 * ggml_type_size(type) / ggml_blck_size(type);
}

// value for a type, or the value of the measured type with the nearest bits per weight
static double hw_lookup(const std::map<std::string, double> & m, ggml_type type) {
    auto it = m.find(ggml_type_name(type));
    if (it != m.end()) {
        return it->second;
    }
    const double bpw = hw_bits_per_weight(type);
    double best      = 0.0;
    double best_dist = INFINITY;
    for (const auto & [name, val] : m) {
        const ggml_type t = hw_type_by_name(name);
        if (t == GGML_TYPE_COUNT || val <= 0.0) {
            continue;
        }
        const double dist = std::fabs(hw_bits_per_weight(t) - bpw);
        if (dist < best_dist) {
            best_dist = dist;
            best      = val;
        }
    }
    return best;
}

double common_hw_dev_profile::gemv_gbps_for(ggml_type type) const {
    return hw_lookup(gemv_gbps, type);
}

double common_hw_dev_profile::gemm_tflops_for(ggml_type type) const {
    return hw_lookup(gemm_tflops, type);
}

const common_hw_dev_profile * common_hw_profile::find(const std::string & dev_name) const {
    for (const auto & d : devs) {
        if (d.name == dev_name) {
            return &d;
        }
    }
    return nullptr;
}

const common_hw_dev_profile * common_hw_profile::cpu() const {
    for (const auto & d : devs) {
        if (d.type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            return &d;
        }
    }
    return nullptr;
}

//
// device selection and key
//

static std::vector<ggml_backend_dev_t> hw_select_devices(const common_hw_profile_params & params) {
    std::vector<ggml_backend_dev_t> ret;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        // ACCEL devices (BLAS, AMX, ...) extend the CPU and have no memory of their own
        if (type == GGML_BACKEND_DEVICE_TYPE_ACCEL || type == GGML_BACKEND_DEVICE_TYPE_META) {
            continue;
        }
        if (type != GGML_BACKEND_DEVICE_TYPE_CPU && !params.dev_names.empty()) {
            const std::string name = ggml_backend_dev_name(dev);
            if (std::find(params.dev_names.begin(), params.dev_names.end(), name) == params.dev_names.end()) {
                continue;
            }
        }
        ret.push_back(dev);
    }
    return ret;
}

static void hw_fnv1a(uint64_t & h, const std::string & s) {
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ull;
    }
    h ^= 0xff; // separator, so that "ab"+"c" != "a"+"bc"
    h *= 0x100000001b3ull;
}

std::string common_hw_profile_key(const common_hw_profile_params & params) {
    uint64_t h = 0xcbf29ce484222325ull;
    hw_fnv1a(h, "v1");
    hw_fnv1a(h, std::to_string(params.n_threads));
    hw_fnv1a(h, std::to_string(params.gemm_batch));
    for (ggml_type t : params.types) {
        hw_fnv1a(h, ggml_type_name(t));
    }
    for (ggml_backend_dev_t dev : hw_select_devices(params)) {
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        hw_fnv1a(h, ggml_backend_dev_name(dev));
        hw_fnv1a(h, ggml_backend_dev_description(dev));
        hw_fnv1a(h, std::to_string((int) ggml_backend_dev_type(dev)));
        hw_fnv1a(h, std::to_string(total));
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016" PRIx64, h);
    return buf;
}

//
// measurement
//

static double hw_median(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n/2] : 0.5*(v[n/2 - 1] + v[n/2]);
}

static void hw_set_threads(ggml_backend_t backend, int32_t n_threads) {
    if (n_threads <= 0) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg == nullptr) {
        return;
    }
    auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn) {
        fn(backend, n_threads);
    }
}

// fill a weight tensor with valid random data of its type, a block of rows is made once and repeated
static bool hw_fill_weight(ggml_tensor * w) {
    const ggml_type type   = w->type;
    const int64_t   n_cols = w->ne[0];
    const int64_t   n_rows = w->ne[1];
    const int64_t   n_blk  = std::min<int64_t>(64, n_rows);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> src(n_cols*n_blk);
    for (float & v : src) {
        v = dist(rng);
    }

    const size_t row_size = ggml_row_size(type, n_cols);
    std::vector<uint8_t> blk(row_size*n_blk);
    if (type == GGML_TYPE_F32) {
        memcpy(blk.data(), src.data(), blk.size());
    } else {
        if (ggml_quantize_requires_imatrix(type)) {
            return false;
        }
        ggml_quantize_chunk(type, src.data(), blk.data(), 0, n_blk, n_cols, nullptr);
    }

    for (int64_t r = 0; r < n_rows; r += n_blk) {
        const int64_t nr = std::min(n_blk, n_rows - r);
        ggml_backend_tensor_set(w, blk.data(), r*row_size, nr*row_size);
    }
    return true;
}

// median seconds for one matmul of a [n_cols, n_rows] weight with a [n_cols, n_tok] activation, < 0 if unsupported
static double hw_time_matmul(ggml_backend_t backend, ggml_type type, int64_t n_cols, int64_t n_rows, int64_t n_tok, int32_t n_rep) {
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return -1.0;
    }

    ggml_tensor * w = ggml_new_tensor_2d(ctx, type, n_cols, n_rows);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tok);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    double ret = -1.0;
    ggml_backend_buffer_t buf = nullptr;

    if (!ggml_backend_supports_op(backend, y)) {
        goto done;
    }
    buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        goto done;
    }
    if (!hw_fill_weight(w)) {
        goto done;
    }
    {
        std::vector<float> xd(ggml_nelements(x));
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (float & v : xd) {
            v = dist(rng);
        }
        ggml_backend_tensor_set(x, xd.data(), 0, ggml_nbytes(x));

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);

        // warmup, also lets backends repack or upload once
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            goto done;
        }
        std::vector<double> t;
        for (int32_t i = 0; i < n_rep; i++) {
            const int64_t t0 = ggml_time_us();
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            t.push_back((ggml_time_us() - t0)*1e-6);
        }
        ret = std::max(hw_median(t), 1e-9);
    }

done:
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    return ret;
}

static void hw_measure_copies(ggml_backend_dev_t dev, ggml_backend_t backend, const common_hw_profile_params & params, common_hw_dev_profile & out) {
    const size_t n_bytes = params.copy_bytes;

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*2,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, n_bytes);
    ggml_backend_buffer_t dbuf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (dbuf == nullptr) {
        ggml_free(ctx);
        return;
    }

    // a pinned host buffer gives the speed that async uploads reach
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    ggml_backend_buffer_t      hbuf      = host_buft ? ggml_backend_buft_alloc_buffer(host_buft, n_bytes) : nullptr;
    std::vector<uint8_t>       pageable;
    uint8_t * src = nullptr;
    if (hbuf) {
        src = (uint8_t *) ggml_backend_buffer_get_base(hbuf);
        out.h2d_pinned = true;
    } else {
        pageable.resize(n_bytes);
        src = pageable.data();
    }
    memset(src, 0x5a, n_bytes);

    std::vector<double> th2d;
    std::vector<double> td2h;
    ggml_backend_tensor_set(t, src, 0, n_bytes); // warmup
    for (int32_t i = 0; i < params.n_rep; i++) {
        int64_t t0 = ggml_time_us();
        ggml_backend_tensor_set(t, src, 0, n_bytes);
        th2d.push_back((ggml_time_us() - t0)*1e-6);

        t0 = ggml_time_us();
        ggml_backend_tensor_get(t, src, 0, n_bytes);
        td2h.push_back((ggml_time_us() - t0)*1e-6);
    }
    out.h2d_gbps = n_bytes / std::max(hw_median(th2d), 1e-9) / 1e9;
    out.d2h_gbps = n_bytes / std::max(hw_median(td2h), 1e-9) / 1e9;

    if (hbuf) {
        ggml_backend_buffer_free(hbuf);
    }
    ggml_backend_buffer_free(dbuf);
    ggml_free(ctx);
}

bool common_hw_profile_measure(const common_hw_profile_params & params, common_hw_profile & out) {
    out = common_hw_profile();
    out.key        = common_hw_profile_key(params);
    out.n_threads  = params.n_threads;
    out.gemm_batch = params.gemm_batch;

    const int64_t n_cols = 4096;

    for (ggml_backend_dev_t dev : hw_select_devices(params)) {
        common_hw_dev_profile dp;
        dp.name        = ggml_backend_dev_name(dev);
        dp.description = ggml_backend_dev_description(dev);
        dp.type        = ggml_backend_dev_type(dev);
        {
            size_t free  = 0;
            size_t total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            dp.mem_total = total;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            LOG_WRN("%s: failed to init device %s, skipping\n", __func__, dp.name.c_str());
            continue;
        }
        hw_set_threads(backend, params.n_threads);

        if (!dp.is_host()) {
            hw_measure_copies(dev, backend, params, dp);
        }

        for (ggml_type type : params.types) {
            const size_t  row_size = ggml_row_size(type, n_cols);
            int64_t       n_rows   = std::max<int64_t>(64, params.gemv_bytes / row_size);
            n_rows -= n_rows % 64;

            const double tv = hw_time_matmul(backend, type, n_cols, n_rows, 1, params.n_rep);
            if (tv > 0.0) {
                dp.gemv_gbps[ggml_type_name(type)] = n_rows*row_size / tv / 1e9;
            }

            // the batched test uses a fixed square weight, its cost grows with the batch and not the cache size
            if (params.gemm_batch > 1) {
                const int64_t n_rows_mm = std::min<int64_t>(n_rows, 4096);
                const double  tm        = hw_time_matmul(backend, type, n_cols, n_rows_mm, params.gemm_batch, params.n_rep);
                if (tm > 0.0) {
                    dp.gemm_tflops[ggml_type_name(type)] = 2.0*n_cols*n_rows_mm*params.gemm_batch / tm / 1e12;
                }
            }
        }

        ggml_backend_free(backend);

        LOG_INF("%s: %s (%s): h2d %.2f GB/s, d2h %.2f GB/s, %zu gemv types\n", __func__,
            dp.name.c_str(), dp.description.c_str(), dp.h2d_gbps, dp.d2h_gbps, dp.gemv_gbps.size());
        out.devs.push_back(std::move(dp));
    }

    return !out.devs.empty();
}

//
// serialization
//

static common_json hw_map_to_json(const std::map<std::string, double> & m) {
    common_json j = common_json::object();
    for (const auto & [k, v] : m) {
        j[k] = v;
    }
    return j;
}

static std::map<std::string, double> hw_map_from_json(const common_json & j) {
    std::map<std::string, double> m;
    if (!j.is_object()) {
        return m;
    }
    for (const auto & [k, v] : j.items()) {
        m[k] = v.get<double>();
    }
    return m;
}

std::string common_hw_profile_to_json(const common_hw_profile & prof) {
    common_json j = common_json::object();
    j["version"]    = prof.version;
    j["key"]        = prof.key;
    j["n_threads"]  = prof.n_threads;
    j["gemm_batch"] = prof.gemm_batch;
    common_json devs = common_json::array();
    for (const auto & d : prof.devs) {
        common_json jd = common_json::object();
        jd["name"]        = d.name;
        jd["description"] = d.description;
        jd["type"]        = d.type;
        jd["mem_total"]   = d.mem_total;
        jd["h2d_gbps"]    = d.h2d_gbps;
        jd["d2h_gbps"]    = d.d2h_gbps;
        jd["h2d_pinned"]  = d.h2d_pinned;
        jd["gemv_gbps"]   = hw_map_to_json(d.gemv_gbps);
        jd["gemm_tflops"] = hw_map_to_json(d.gemm_tflops);
        devs.push_back(jd);
    }
    j["devs"] = devs;
    return j.dump(2);
}

bool common_hw_profile_from_json(const std::string & text, common_hw_profile & out) {
    common_json j = common_json::parse_no_throw(text);
    if (j.is_discarded() || !j.is_object()) {
        return false;
    }
    try {
        common_hw_profile p;
        p.version = j.value("version", 0);
        if (p.version != 1) {
            return false;
        }
        p.key        = j.value("key", std::string());
        p.n_threads  = j.value("n_threads", 0);
        p.gemm_batch = j.value("gemm_batch", 0);
        if (j.contains("devs") && j.at("devs").is_array()) {
            const common_json & devs = j.at("devs");
            for (size_t i = 0; i < devs.size(); i++) {
                const common_json & jd = devs.at(i);
                common_hw_dev_profile d;
                d.name        = jd.value("name", std::string());
                d.description = jd.value("description", std::string());
                d.type        = jd.value("type", 0);
                d.mem_total   = jd.value("mem_total", (unsigned long long) 0);
                d.h2d_gbps    = jd.value("h2d_gbps", 0.0);
                d.d2h_gbps    = jd.value("d2h_gbps", 0.0);
                d.h2d_pinned  = jd.value("h2d_pinned", false);
                if (jd.contains("gemv_gbps")) {
                    d.gemv_gbps = hw_map_from_json(jd.at("gemv_gbps"));
                }
                if (jd.contains("gemm_tflops")) {
                    d.gemm_tflops = hw_map_from_json(jd.at("gemm_tflops"));
                }
                p.devs.push_back(std::move(d));
            }
        }
        out = std::move(p);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

std::string common_hw_profile_default_path(const std::string & key) {
    return (fs_get_cache_directory() / ("hw-profile-" + key + ".json")).string();
}

bool common_hw_profile_save(const std::string & path, const common_hw_profile & prof) {
    std::error_code ec;
    const std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path(), ec);
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }
    f << common_hw_profile_to_json(prof);
    return (bool) f;
}

bool common_hw_profile_load(const std::string & path, common_hw_profile & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return common_hw_profile_from_json(ss.str(), out);
}

bool common_hw_profile_get(const common_hw_profile_params & params, bool refresh, common_hw_profile & out) {
    const std::string key  = common_hw_profile_key(params);
    const std::string path = common_hw_profile_default_path(key);
    if (!refresh && common_hw_profile_load(path, out) && out.key == key) {
        LOG_INF("%s: using cached hardware profile %s\n", __func__, path.c_str());
        return true;
    }
    LOG_INF("%s: measuring hardware, this takes a few seconds\n", __func__);
    if (!common_hw_profile_measure(params, out)) {
        return false;
    }
    if (!common_hw_profile_save(path, out)) {
        LOG_WRN("%s: failed to save hardware profile to %s\n", __func__, path.c_str());
    }
    return true;
}

bool common_hw_profile_resolve(const std::string & profile_path, int32_t n_threads, common_hw_profile & out) {
    if (!profile_path.empty()) {
        if (!common_hw_profile_load(profile_path, out)) {
            LOG_ERR("%s: failed to load hardware profile %s\n", __func__, profile_path.c_str());
            return false;
        }
        return true;
    }
    common_hw_profile_params hp;
    hp.n_threads = n_threads;
    return common_hw_profile_get(hp, false, out);
}

bool common_hw_profile_offload_params(const common_hw_profile & prof, const std::string & dev_name, ggml_backend_offload_params & out) {
    const common_hw_dev_profile * cpu = prof.cpu();
    const common_hw_dev_profile * dev = prof.find(dev_name);
    if (cpu == nullptr || dev == nullptr || dev->is_host() || dev->h2d_gbps <= 0.0) {
        return false;
    }
    out = {};
    out.h2d_gbps = (float) dev->h2d_gbps;
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const ggml_type type = (ggml_type) t;
        if (ggml_type_size(type) == 0 || ggml_blck_size(type) == 0) {
            continue; // removed types
        }
        out.host_gemv_gbps  [t] = (float) cpu->gemv_gbps_for(type);
        out.host_gemm_tflops[t] = (float) cpu->gemm_tflops_for(type);
        out.dev_gemv_gbps   [t] = (float) dev->gemv_gbps_for(type);
        out.dev_gemm_tflops [t] = (float) dev->gemm_tflops_for(type);
    }
    return true;
}

int common_hw_profile_apply_offload(const common_hw_profile & prof) {
    int n = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        ggml_backend_offload_params params;
        if (!common_hw_profile_offload_params(prof, ggml_backend_dev_name(dev), params)) {
            LOG_WRN("%s: hardware profile has no entry for %s, it keeps the fixed offload rule\n", __func__, ggml_backend_dev_name(dev));
            continue;
        }
        ggml_backend_dev_set_offload_params(dev, &params);
        n++;
    }
    return n;
}
