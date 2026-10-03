# Hybrid inference (lamb.cpp)

lamb.cpp runs a model across VRAM, system RAM, GPU and CPU cores as one pool. This page covers the parts that exist today.

## Hardware profile

`llama-hw-profile` measures the devices on this machine:

- host -> device and device -> host copy speed (from a pinned host buffer when the device has one)
- single-token matmul speed, in GB/s of weights read, for f16, q8_0 and q4_K
- batched matmul throughput, in TFLOPS, at a batch of 512

```sh
llama-hw-profile -t 16            # measure once, cache, print as JSON
llama-hw-profile -t 16 -r         # measure again
llama-hw-profile -o prof.json     # write to a given file
```

The profile is cached in the llama.cpp cache directory (`LLAMA_CACHE`) as `hw-profile-<key>.json`. The key changes when the devices, their memory size, the thread count or the measured types change. Types that were not measured use the measured type with the nearest bits per weight.

## Throughput estimate of a placement

`--fit-estimate on` loads the model with `no_alloc` after `--fit` (or after your own `-ngl` / `-ot`), reads where every weight tensor was placed and prints a predicted decode and prefill speed:

```sh
llama-fit-params -m model.gguf --fit-estimate on -t 16
llama-server -m model.gguf -ngl 99 -ncmoe 20 --fit off --fit-estimate on --hw-profile prof.json
```

The cost model counts:

- decode: every dense weight once per token, `n_expert_used / n_expert` of each routed expert stack, one row of the token embedding, and `ctx_fill` (0.5) of the context memory (KV cache or recurrent state, as measured by the memory module, so MLA and SSM layers are counted at their real size)
- prefill: the matmul FLOPs of one ubatch on the device that holds the weight; host weights are copied to the first GPU when `n_ubatch >= GGML_OP_OFFLOAD_MIN_BATCH` (default 32), as `ggml-backend` does, and only the experts that the ubatch uses are copied
- a fixed cost (20 us) for each change of device along the forward pass

It does not count attention compute, sampling or graph overhead, so it is an upper bound. Use it to compare placements, not to promise a number.

Why the existing `--fit` already places MoE models well: per byte of VRAM, a dense tensor saves `n_expert / n_expert_used` times more decode time than a routed expert stack (16x for 128 experts with 8 used), so dense tensors go to the GPU first and experts fill what is left. `tests/test-hybrid-plan.cpp` checks this with the cost model.

## Cost based op offload (`--offload-policy cost`)

With weights in system memory, `ggml-backend` runs a matmul on the GPU when the GPU wants to "offload" it: the weight is copied over PCIe and the op runs there. Upstream decides this with a fixed batch size (`GGML_OP_OFFLOAD_MIN_BATCH`, default 32), the same for every op.

That rule is wrong in both directions:

- a dense weight can pay for the copy at a smaller batch on a fast link
- a routed expert stack needs a much larger batch: with 128 experts and 8 used, a batch of 32 tokens already reads about 87% of the experts, so most of the stack is copied, while the CPU only computes 8 of 128 experts per token

`--offload-policy cost` compares, per op, `max(read / host_gemv, flops / host_tflops)` on the host with `read / h2d + max(read / dev_gemv, flops / dev_tflops)` on the GPU, where `read` is the bytes of the experts the batch uses (`1 - (1 - n_used / n_expert) ^ n_tokens` of the stack). The numbers come from the hardware profile. The decision sits in `ggml_backend_dev_offload_op`, so it applies to every backend; ops other than matmuls and devices without a profile keep the backend rule, and setting `GGML_OP_OFFLOAD_MIN_BATCH` in the environment always restores it.

With the mock 12 GB PCIe4 profile of the tests, a 4096x4096 q4_K matmul is offloaded from 16 tokens and a 128-expert stack from 256 tokens; at `-ub 64` the estimate for a 30B-A3B with experts in RAM goes from 47 to 118 t/s prefill.

```sh
llama-bench -m moe.gguf -ncmoe 99 -p 64,128,512,2048 -n 0 -opol fixed,cost
llama-server -m moe.gguf -ncmoe 99 --offload-policy cost
```

`--fit-estimate` follows the policy that is set, so it predicts both.

## Weight prefetch (`GGML_SCHED_PREFETCH=1`)

When an op with weights in system memory runs on the GPU, the scheduler gives it its own split and copies the weights in before the split runs. Upstream does that copy on the same stream as the compute, so the GPU waits for PCIe before every such split and PCIe waits for the GPU between them.

With `GGML_SCHED_PREFETCH=1`:

- the weight copies of each split go into one of two staging slots of the GPU instead of the compute buffer; the slots alternate between consecutive weight splits of a device
- a second backend instance of the same device (a second CUDA stream) uploads the weights of the next weight split while the current one computes
- two events per slot keep the order: a split waits until its slot is `ready`, and an upload into a slot waits until the split that read it before has `freed` it

Left on the normal path, so nothing changes for them: weights that another split also reads (a slot is reused two splits later), `MUL_MAT_ID` weights at the start of a split (they are copied one used expert at a time after the router ran, which needs the expert ids), pipeline parallelism (`n_copies > 1`), and backends without events, async uploads or a second instance.

The staging slots cost `2 x` the largest set of weights one split copies, and are reported in the compute buffer size of the device (`ggml_backend_sched_get_buffer_size`). The upload from host memory overlaps best when the host weights are in pinned memory (`--load-mode none` uses the pinned host buffer type of the GPU); from mmap'd pages CUDA stages the copy and the host thread waits for it, but the previous split still runs on the GPU meanwhile.

It matters for prefill with weights in system memory, which M2 makes more frequent for dense weights. For decode no weights are copied, so it does nothing there.

```sh
GGML_SCHED_PREFETCH=1 llama-bench -m model.gguf -ngl 10 -p 512,2048 -n 0
```

`tests/test-sched-prefetch.cpp` runs a graph with dense, quantized, shared and MoE weights in host memory on a mock asynchronous GPU (`tests/mock-gpu-backend.cpp`: one worker thread per stream, real events, slow uploads, buffers filled with NaN until written) and requires bit-identical results to a CPU-only run, an upload count that shows the staging was used, and copy time that overlapped compute. Removing either of the two waits makes it fail.

## MoE expert cache (`--expert-cache N`)

With routed experts in system memory (`-cmoe`, `-ncmoe`, `-ot exps=CPU`), every decoded token computes `n_expert_used` experts per layer on the CPU. Expert use is skewed, so a few experts per layer serve a large part of the tokens. `--expert-cache N` keeps the N most used experts of each such layer in the memory of the GPU that runs the layer, and computes them there (Fiddler 2402.07033, HybriMoE 2504.05897, 2512.16473, CoX-MoE 2605.17889).

How it runs, without new ggml ops or kernel changes:

- per cached layer there are three small lookup tables, read with `GET_ROWS` on the selected experts: the cache slot of each expert, the expert the host computes, and a hit/miss mask
- `build_moe_ffn` runs the expert FFN twice: once on the cache stacks `[n_embd, n_ff, N]` in VRAM with the slot ids, and once on the host stacks with the host ids
- on a hit the host computes one shared stand-in expert (the most used uncached one, which a miss is likely to read anyway), so the host reads the missed experts and at most one more; on a miss the GPU computes slot 0 and the result is dropped
- the two results are combined as `dev * hit + host * miss`; `x * 1 + y * 0` is exact, so the output equals the uncached one bit for bit when both paths compute the same values
- after each ubatch the selected experts are read back and counted; every `LLAMA_EXPERT_CACHE_INTERVAL` ubatches (default 8) the most used experts replace the least used cached ones (at most 64 uploads per update, a swap needs a 25% higher count, counts halve each update so the cache follows the text)

Cost: `N x` the size of one expert of every cached layer in VRAM, logged at context creation (for a 48-layer 30B-A3B in q4_K about 128 MiB per slot, so 32 slots are about 4 GiB). This memory is not yet counted by `--fit`, so leave room for it with `--fit-target`. Each ubatch also synchronizes once to read the expert ids.

What is not cached: layers whose stacks are repacked for the CPU (use `--no-repack`), layers on a CPU-only device, models with an expert LoRA loaded, and architectures that do not route experts through `build_moe_ffn`.

The hit rate is logged when the context is freed. It decides the gain: the host reads `misses + 1` experts per layer instead of `n_expert_used`.

```sh
llama-bench -m moe.gguf -ncmoe 99 -n 128 -p 0 -ec 0,16,32
llama-server -m moe.gguf -ncmoe 99 --expert-cache 32
```

`tests/test-expert-cache.cpp` runs every generated MoE test model with and without a cache placed in host memory (`LLAMA_EXPERT_CACHE_HOST=1`) and updated after every ubatch, and requires bit-identical logits over a prompt and 32 decoded tokens: 55 models, 52 of them route through the cache. Uploading the wrong expert into a slot makes all 52 fail.

## LoRA cache (`--lora-cache MiB`)

Upstream loads every LoRA adapter into the memory of the device that holds the weight it changes, so N adapters on GPU layers cost N times their size in VRAM, whether or not a request uses them. `--lora-cache MiB` (S-LoRA 2311.03285) changes where adapters live:

- adapters loaded after the flag keep a home copy in system memory: the pinned host buffer of the GPU when it has one, else plain system memory
- a model-level cache copies the adapters that a context applies (`llama_set_adapters_lora`, the server's per-request `lora` field) to the GPU on a worker thread, within the budget per GPU
- until the copy is done the graph reads the home copy, so that adapter's matmuls run on the CPU (or are offloaded by the scheduler for large batches) instead of making the request wait
- when a copy is done, the next decode of every context sees a new generation of the cache and rebuilds its graph with the device copy
- an adapter that no context applies is evicted first, least recently used; an adapter that a context applies is never evicted, and an adapter that does not fit stays on the CPU path

A context synchronizes before it changes its adapters, so a copy is never freed while one of its graphs runs. Contexts that share a model across threads must not change adapters while another thread builds a graph: the cache swaps the tensor pointers of the shared adapter.

```sh
llama-server -m model.gguf -ngl 99 --lora-cache 512 --lora a.gguf --lora b.gguf --lora c.gguf ...
```

Not done: picking the CPU path on purpose for small adapters (rank <= 16) when the matmul there is cheaper than the copy, and a shared-basis format (VeRA 2310.11454) that would make each adapter a few KiB.

`tests/test-lora-cache.cpp` writes three random rank-4 adapters for a generated test model, loads them into a plain model and into one with a cache that fits two (in host memory, `LLAMA_LORA_CACHE_HOST=1`), and switches between them 8 times, half of the time while the upload is still in flight. The logits must be bit-identical to the plain model each time and differ from no adapter, the cache must upload and evict and stay within its budget. Leaving the B matrix of the device copy unfilled makes all 8 steps fail.

## Concurrent host and device work (`GGML_SCHED_CONCURRENT=1`)

The scheduler runs splits in graph order. A host split is computed on the calling thread, so while the CPU computes routed experts the GPU waits, even for work that does not need the host result. In a MoE layer that work exists: the shared expert, and with `--expert-cache` the cached experts.

With `GGML_SCHED_CONCURRENT=1`, before a host split runs the scheduler looks at the next device split, finds its leading nodes that read nothing the host split makes, copies the inputs they need and starts them on the device. The host split then runs while the device works, and the rest of the device split (the node that joins both results) runs afterwards as usual. KTransformers (SOSP'25) gets its CPU/GPU overlap the same way, without deferring experts to the next layer, so the output does not change.

It needs the device part to come after the host part in the graph: `build_moe_ffn` builds the host experts before the cached ones for this reason. The memory plan of ggml-alloc stays valid: the host split's inputs are copied before the device starts, and every device tensor the early nodes use was planned to be live at that point anyway.

Not done: weight inputs (the expert-wise copy of `MUL_MAT_ID` weights keeps its order), pipeline parallelism (`n_copies > 1`), and graphs with an eval callback. In the model code the shared expert is built after the routed experts and joined with them by an add, so it should form the leading nodes of the device split after the host experts and overlap with them; this is not verified on a real model here, `GGML_SCHED_DEBUG=2` prints the splits to check it.

`tests/test-sched-concurrent.cpp` builds the MoE shape (device, slow host op, independent device work, join) on the mock GPU with a device time per node: without the flag the device never runs during the host op, with it the device does, the run takes 304 ms instead of 424 ms (the predicted 100 ms instead of 140 ms per run), and the results are bit-identical to a CPU-only run. Making the dependency check ignore host results lets the join run early and the results differ.

## Tests

```sh
ctest --test-dir build -R "test-hw-profile|test-hybrid-plan|test-offload-policy|test-sched-prefetch|test-expert-cache|test-lora-cache|test-sched-concurrent" --output-on-failure
```

`test-hybrid-plan` also loads every generated test model (`test-generate-models`) with `no_alloc` and checks the collected placement.

## Status

| milestone | state |
|---|---|
| M0 hardware profile | done, CPU verified, GPU copy and matmul numbers not yet measured on a GPU |
| M1 cost model and `--fit-estimate` | done, accuracy against real runs not yet measured on a GPU |
| M2 cost based op offload | done, CPU verified, gain not yet measured on a GPU |
| M3 weight prefetch on a second stream | done, verified on a mock async GPU, not yet measured on CUDA |
| M4 MoE expert cache, CPU compute on miss | done, bit-exact on 52 MoE architectures on the CPU, gain not yet measured on a GPU |
| M7 LoRA cache, CPU path until the copy is done | done, bit-exact on the CPU, not yet measured on a GPU |
| M5 concurrent host and device splits | done, verified on a mock async GPU, not yet measured on CUDA |

To check the model on a GPU machine, compare `--fit-estimate` with `llama-bench -m model.gguf -fa 1 -p 512 -n 128` for the same placement.
