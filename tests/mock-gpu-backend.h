#pragma once

// A test-only device that behaves like an asynchronous GPU without being one:
//   - its buffers are not host buffers, so the scheduler copies inputs to it
//   - each backend instance is a stream: a worker thread that runs copies and graphs in order
//   - events order work between streams, as CUDA events do
//   - copies can be slowed down, so a missing wait reads stale data and gives a wrong result
// Graphs are computed with the CPU backend on the buffer memory.

#include "ggml-backend.h"

#include <cstdint>

struct mock_gpu_stats {
    int64_t n_set_async   = 0; // async uploads, on any instance
    int64_t n_graphs      = 0; // graph computes
    int64_t n_instances   = 0; // backend instances created
    int64_t n_overlap_us  = 0; // time a copy ran while a graph ran on another instance of the device
};

// a device that is not registered globally, free it with mock_gpu_dev_free
//   copy_delay_us: sleep this long in each async upload
//   offload_min_batch: ops with weights on the host are offloaded from this batch size
ggml_backend_dev_t mock_gpu_dev_new(int64_t copy_delay_us, int64_t offload_min_batch);
void               mock_gpu_dev_free(ggml_backend_dev_t dev);

mock_gpu_stats mock_gpu_get_stats(ggml_backend_dev_t dev);
void           mock_gpu_reset_stats(ggml_backend_dev_t dev);

// sleep this long per node of each graph compute, to stand for device work
void mock_gpu_set_compute_delay(ggml_backend_dev_t dev, int64_t us);

// number of graphs running on the device right now
int mock_gpu_running_graphs(ggml_backend_dev_t dev);
