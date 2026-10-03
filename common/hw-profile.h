#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Measured throughput of the devices on this machine.
// The hybrid planner and the offload policy read these numbers to decide where weights live and where ops run.

struct common_hw_dev_profile {
    std::string name;
    std::string description;
    int32_t     type      = 0; // enum ggml_backend_dev_type
    uint64_t    mem_total = 0;

    // host <-> device copy speed in GB/s, 0 for host devices
    double h2d_gbps   = 0.0;
    double d2h_gbps   = 0.0;
    bool   h2d_pinned = false; // true if the copy source was a pinned host buffer

    // ggml type name -> weight bytes read per second during a single-token matmul, in GB/s
    std::map<std::string, double> gemv_gbps;

    // ggml type name -> throughput of a batched matmul, in TFLOPS
    std::map<std::string, double> gemm_tflops;

    bool is_host() const;

    // speed for a type, uses the measured type with the nearest bits per weight if this type was not measured
    double gemv_gbps_for (enum ggml_type type) const;
    double gemm_tflops_for(enum ggml_type type) const;
};

struct common_hw_profile {
    int32_t     version   = 1;
    std::string key;       // identifies the hardware and settings the numbers belong to
    int32_t     n_threads = 0;
    int32_t     gemm_batch = 0;

    std::vector<common_hw_dev_profile> devs; // the CPU entry has type GGML_BACKEND_DEVICE_TYPE_CPU

    const common_hw_dev_profile * find(const std::string & dev_name) const;
    const common_hw_dev_profile * cpu() const;
};

struct common_hw_profile_params {
    int32_t                    n_threads   = 0;   // <= 0: use the CPU backend default
    size_t                     gemv_bytes  = 256ull*1024*1024; // weight size per matmul test, larger than the CPU cache
    size_t                     copy_bytes  = 256ull*1024*1024; // size of each host <-> device copy
    int32_t                    gemm_batch  = 512;
    int32_t                    n_rep       = 5;   // timed runs per number, the median is kept
    std::vector<enum ggml_type> types      = {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K};
    std::vector<std::string>   dev_names;         // empty: all devices
};

// key of the current machine for the given settings, stable across runs of the same build
std::string common_hw_profile_key(const common_hw_profile_params & params);

// measure all selected devices, returns false if no device could be measured
bool common_hw_profile_measure(const common_hw_profile_params & params, common_hw_profile & out);

std::string common_hw_profile_to_json  (const common_hw_profile & prof);
bool        common_hw_profile_from_json(const std::string & text, common_hw_profile & out);

// default cache file for a key, in the llama.cpp cache directory
std::string common_hw_profile_default_path(const std::string & key);

bool common_hw_profile_save(const std::string & path, const common_hw_profile & prof);
bool common_hw_profile_load(const std::string & path, common_hw_profile & out);

// load the cached profile for this machine if the key matches, else measure and save it
bool common_hw_profile_get(const common_hw_profile_params & params, bool refresh, common_hw_profile & out);
