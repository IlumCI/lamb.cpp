// Tests for the hardware profile: measurement on the CPU, lookup fallback, JSON round trip and the cache key.

#include "hw-profile.h"

#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

static common_hw_profile_params small_params() {
    common_hw_profile_params p;
    p.n_threads  = 2;
    p.gemv_bytes = 8ull*1024*1024;
    p.copy_bytes = 4ull*1024*1024;
    p.gemm_batch = 16;
    p.n_rep      = 2;
    p.types      = {GGML_TYPE_F16, GGML_TYPE_Q4_K};
    return p;
}

static void test_measure_cpu() {
    common_hw_profile prof;
    CHECK(common_hw_profile_measure(small_params(), prof));

    const common_hw_dev_profile * cpu = prof.cpu();
    CHECK(cpu != nullptr);
    if (cpu == nullptr) {
        return;
    }
    CHECK(cpu->is_host());
    CHECK(cpu->h2d_gbps == 0.0);
    CHECK(cpu->gemv_gbps_for(GGML_TYPE_F16)  > 0.0);
    CHECK(cpu->gemv_gbps_for(GGML_TYPE_Q4_K) > 0.0);
    CHECK(cpu->gemm_tflops_for(GGML_TYPE_Q4_K) > 0.0);
    CHECK(std::isfinite(cpu->gemv_gbps_for(GGML_TYPE_F16)));
    CHECK(prof.find(cpu->name) == cpu);
    CHECK(prof.find("no-such-device") == nullptr);
    printf("cpu: %s, gemv f16 %.2f GB/s, q4_K %.2f GB/s, gemm q4_K %.3f TFLOPS\n", cpu->description.c_str(),
        cpu->gemv_gbps_for(GGML_TYPE_F16), cpu->gemv_gbps_for(GGML_TYPE_Q4_K), cpu->gemm_tflops_for(GGML_TYPE_Q4_K));
}

static void test_lookup_fallback() {
    common_hw_dev_profile d;
    d.gemv_gbps["f16"]  = 20.0;
    d.gemv_gbps["q4_K"] = 40.0;

    CHECK(d.gemv_gbps_for(GGML_TYPE_F16)  == 20.0);
    CHECK(d.gemv_gbps_for(GGML_TYPE_Q4_K) == 40.0);
    // q4_0 has the same bits per weight as q4_K, bf16 the same as f16
    CHECK(d.gemv_gbps_for(GGML_TYPE_Q4_0) == 40.0);
    CHECK(d.gemv_gbps_for(GGML_TYPE_BF16) == 20.0);
    // f32 is nearer to f16 than to q4_K
    CHECK(d.gemv_gbps_for(GGML_TYPE_F32)  == 20.0);
    // nothing measured gives 0
    CHECK(d.gemm_tflops_for(GGML_TYPE_F16) == 0.0);
}

static common_hw_profile synthetic_profile() {
    common_hw_profile p;
    p.key        = "0123456789abcdef";
    p.n_threads  = 8;
    p.gemm_batch = 512;

    common_hw_dev_profile cpu;
    cpu.name        = "CPU";
    cpu.description = "Test CPU";
    cpu.type        = GGML_BACKEND_DEVICE_TYPE_CPU;
    cpu.mem_total   = 64ull*1024*1024*1024;
    cpu.gemv_gbps["q4_K"]   = 45.5;
    cpu.gemm_tflops["q4_K"] = 0.75;

    common_hw_dev_profile gpu;
    gpu.name        = "CUDA0";
    gpu.description = "Test GPU";
    gpu.type        = GGML_BACKEND_DEVICE_TYPE_GPU;
    gpu.mem_total   = 12ull*1024*1024*1024;
    gpu.h2d_gbps    = 24.5;
    gpu.d2h_gbps    = 25.25;
    gpu.h2d_pinned  = true;
    gpu.gemv_gbps["q4_K"]   = 450.0;
    gpu.gemm_tflops["q4_K"] = 60.0;

    p.devs = {cpu, gpu};
    return p;
}

static void test_json_round_trip() {
    const common_hw_profile p = synthetic_profile();
    const std::string j1 = common_hw_profile_to_json(p);

    common_hw_profile q;
    CHECK(common_hw_profile_from_json(j1, q));
    CHECK(common_hw_profile_to_json(q) == j1);
    CHECK(q.devs.size() == 2);
    CHECK(q.key == p.key);
    CHECK(q.devs[1].h2d_pinned);
    CHECK(q.devs[1].mem_total == p.devs[1].mem_total);
    CHECK(q.devs[1].gemv_gbps_for(GGML_TYPE_Q4_K) == 450.0);
    CHECK(q.cpu() == &q.devs[0]);

    common_hw_profile r;
    CHECK(!common_hw_profile_from_json("not json", r));
    CHECK(!common_hw_profile_from_json("{\"version\": 99}", r));
    CHECK(!common_hw_profile_from_json("[1, 2]", r));
}

static void test_save_load() {
    const common_hw_profile p = synthetic_profile();
    const std::string path = (std::filesystem::temp_directory_path() / "test-hw-profile/sub/prof.json").string();
    CHECK(common_hw_profile_save(path, p));

    common_hw_profile q;
    CHECK(common_hw_profile_load(path, q));
    CHECK(common_hw_profile_to_json(q) == common_hw_profile_to_json(p));

    common_hw_profile r;
    CHECK(!common_hw_profile_load(path + ".missing", r));

    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::temp_directory_path() / "test-hw-profile", ec);
}

static void test_key() {
    common_hw_profile_params a = small_params();
    common_hw_profile_params b = small_params();
    CHECK(common_hw_profile_key(a) == common_hw_profile_key(b));
    CHECK(common_hw_profile_key(a).size() == 16);

    b.n_threads = 3;
    CHECK(common_hw_profile_key(a) != common_hw_profile_key(b));

    b = small_params();
    b.types.push_back(GGML_TYPE_Q8_0);
    CHECK(common_hw_profile_key(a) != common_hw_profile_key(b));

    common_hw_profile prof;
    CHECK(common_hw_profile_measure(a, prof));
    CHECK(prof.key == common_hw_profile_key(a));
}

int main() {
    ggml_backend_load_all();

    test_lookup_fallback();
    test_json_round_trip();
    test_save_load();
    test_key();
    test_measure_cpu();

    if (n_fail > 0) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
