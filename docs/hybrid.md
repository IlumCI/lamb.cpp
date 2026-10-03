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

## Tests

```sh
ctest --test-dir build -R "test-hw-profile|test-hybrid-plan" --output-on-failure
```

`test-hybrid-plan` also loads every generated test model (`test-generate-models`) with `no_alloc` and checks the collected placement.

## Status

| milestone | state |
|---|---|
| M0 hardware profile | done, CPU verified, GPU copy and matmul numbers not yet measured on a GPU |
| M1 cost model and `--fit-estimate` | done, accuracy against real runs not yet measured on a GPU |
| M2 cost based op offload | done, CPU verified, gain not yet measured on a GPU |
| M3 async double-buffered weight streaming | planned |
| M4 GPU expert cache, CPU compute on miss | planned |
| M7 tiered LoRA cache | planned |
| M5 concurrent CPU/GPU splits | planned |

To check the model on a GPU machine, compare `--fit-estimate` with `llama-bench -m model.gguf -fa 1 -p 512 -n 128` for the same placement.
