// Measures host <-> device copy speed and matmul speed per device and caches the result.
// The hybrid placement planner (--fit-mode throughput) reads this profile.

#include "hw-profile.h"
#include "common.h"
#include "log.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void print_usage(const char * prog) {
    printf("usage: %s [options]\n\n", prog);
    printf("options:\n");
    printf("  -t,  --threads N      CPU threads (default: backend default)\n");
    printf("  -d,  --device NAME    measure only this GPU device, can repeat (default: all)\n");
    printf("  -o,  --output PATH    write the profile here (default: cache directory)\n");
    printf("  -r,  --refresh        measure again even if a cached profile exists\n");
    printf("  -s,  --size-mib N     weight and copy size per test in MiB (default: 256)\n");
    printf("  -n,  --reps N         timed runs per number (default: 5)\n");
    printf("  -b,  --batch N        batch size of the matmul throughput test (default: 512)\n");
    printf("  -h,  --help           show this help\n");
}

int main(int argc, char ** argv) {
    common_hw_profile_params params;
    std::string output;
    bool refresh = false;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: missing value for %s\n", arg.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (arg == "-t" || arg == "--threads") {
            params.n_threads = atoi(next());
        } else if (arg == "-d" || arg == "--device") {
            params.dev_names.push_back(next());
        } else if (arg == "-o" || arg == "--output") {
            output = next();
        } else if (arg == "-r" || arg == "--refresh") {
            refresh = true;
        } else if (arg == "-s" || arg == "--size-mib") {
            params.gemv_bytes = params.copy_bytes = (size_t) atoll(next()) * 1024 * 1024;
        } else if (arg == "-n" || arg == "--reps") {
            params.n_rep = atoi(next());
        } else if (arg == "-b" || arg == "--batch") {
            params.gemm_batch = atoi(next());
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    common_init();
    ggml_backend_load_all();

    common_hw_profile prof;
    bool ok = false;
    if (output.empty()) {
        ok = common_hw_profile_get(params, refresh, prof);
        output = common_hw_profile_default_path(prof.key);
    } else {
        ok = common_hw_profile_measure(params, prof) && common_hw_profile_save(output, prof);
    }
    if (!ok) {
        fprintf(stderr, "error: failed to measure or save the hardware profile\n");
        return 1;
    }

    printf("%s\n", common_hw_profile_to_json(prof).c_str());
    fprintf(stderr, "profile: %s\n", output.c_str());
    return 0;
}
