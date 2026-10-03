#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct llama_adapter_lora;
struct llama_adapter_lora_weight;

// A device-memory cache for LoRA adapters whose home copy is in system memory.
// Adapters that a context uses are uploaded on a worker thread, within a byte budget per device buffer type.
// Until the upload is done the graph reads the home copy, so the LoRA matmuls run on the host instead of stalling.
// When an upload is done, poll() points the adapter weights at the device copy and changes the generation,
// which tells every context to rebuild its graph. Adapters that no context uses are evicted first, least recently used.
class llama_lora_cache {
public:
    // budget in bytes per device; allow_host also caches for host buffer types (for tests)
    llama_lora_cache(size_t budget, bool allow_host);
    ~llama_lora_cache();

    size_t budget()     const { return budget_; }
    bool   allow_host() const { return allow_host_; }

    // the adapters a user (a context) applies from now on, replaces its previous set
    // the caller must make sure that no graph of this user that reads an adapter it drops is still running
    void use(const void * user, const std::vector<llama_adapter_lora *> & adapters);

    // apply finished uploads, returns the generation
    uint64_t poll();

    // drop an adapter before it is freed
    void forget(llama_adapter_lora * adapter);

    struct info {
        int64_t n_uploads   = 0; // uploads finished
        int64_t n_evictions = 0;
        int64_t n_failed    = 0; // uploads that could not allocate
        size_t  n_bytes     = 0; // bytes in device memory, including uploads in flight
        int32_t n_resident  = 0; // adapter copies in device memory
    };
    info get_info();

private:
    struct entry {
        llama_adapter_lora *       adapter = nullptr;
        ggml_backend_buffer_type_t buft    = nullptr;
        size_t                     bytes   = 0;

        enum { HOST, UPLOADING, DONE, RESIDENT } state = HOST;

        bool     failed   = false;
        uint64_t last_use = 0;

        ggml_context_ptr        ctx;
        ggml_backend_buffer_ptr buf;

        // device copies of the weights of this buffer type: weight, a, b
        struct dev_weight {
            struct llama_adapter_lora_weight * w;
            ggml_tensor * a;
            ggml_tensor * b;
        };
        std::vector<dev_weight> dev;
    };

    entry & get_entry(llama_adapter_lora * adapter, ggml_backend_buffer_type_t buft);
    int     n_users(const llama_adapter_lora * adapter) const;
    void    point(entry & e, bool to_device);
    bool    make_room(ggml_backend_buffer_type_t buft, size_t bytes);
    void    run();

    const size_t budget_;
    const bool   allow_host_;

    std::mutex              mutex;
    std::condition_variable cv;
    std::condition_variable cv_idle;
    std::deque<entry *>     queue;
    bool                    busy = false;
    bool                    stop = false;
    std::thread             worker;

    std::list<entry>                                             entries;
    std::map<const void *, std::vector<llama_adapter_lora *>>    users;
    std::map<ggml_backend_buffer_type_t, size_t>                 used;
    uint64_t                                                     clock = 0;
    uint64_t                                                     generation = 0;
    info                                                         info_;
};
