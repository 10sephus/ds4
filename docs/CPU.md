# CPU inference on x86-64 Linux

[README](../README.md) | [Getting started](../README.md#start-here)

The CPU backend runs DeepSeek V4 Flash Q2 and V4.1 Flash Q2 and Q4 without a
GPU. It is a reference path kept correct first: the V4 logits are bitwise
those of the scalar code it replaced, every kernel is checked against a scalar
oracle, and the build avoids `-ffast-math` so that sums keep their written
order. On a dual Xeon Gold 6230 with DDR4-2400 it decodes V4 Flash Q2 at
about 10 tokens/s and V4.1 Flash Q2 at about 9 tokens/s, a quarter of the M5
Max in [PERFORMANCE.md](PERFORMANCE.md), which remains the production target.

## Build and run

`make cpu` builds `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval` and `ds4-agent`
for the CPU only. The binaries select the CPU backend with `--cpu`:

```sh
make -j8 cpu
./ds4 --cpu -m ds4flash.gguf --ctx 4096 --nothink --temp 0 -p "Say hello."
```

The main weights are mapped from the GGUF and read every token, so they must
fit in RAM: 81 GiB for V4 Flash Q2, 152 GiB for V4.1 Flash Q2 and 294 GiB for
V4.1 Flash Q4, plus the KV cache. V4.1's 189 GiB of Engram tables stay on
disk and are read per token, so keep the file on a fast local SSD; see the
[model guide](MODELS.md#deepseek-v41-flash) for the downloads.

Vision, speculative decoding and SSD streaming of the experts are not
available with `--cpu`. Only x86-64 has SIMD kernels (AVX2, AVX-512 with VNNI
and F16C); other architectures and older CPUs run the scalar code.

## Threads and NUMA

The pool runs one thread per physical core in the process affinity mask, at
most 64; `-t` or `DS4_THREADS` override it and a larger request is reduced
with a warning. Hyperthreads add nothing to a memory-bound decode.

On a Linux host with two NUMA nodes the threads are split between the nodes
and every matrix is read half from each. `--warm-weights` then migrates each
half to the node that reads it, so the weights stream from local memory;
`--numa-place all|dense|none` chooses whether the routed experts are placed
too (`all`, the default, falls back to `dense` when a node lacks the free
memory), and `DS4_NUMA=0` turns NUMA awareness off. Placement persists in the
page cache, so a second run on the same file starts placed.

## Arithmetic

V4 Flash Q2 keeps the arithmetic of the scalar CPU code bit for bit: the
IQ2_XXS and Q2_K expert dots, the Q8_0 projections and the F16 matrix-vector
products are integer or lane-wise AVX-512 kernels that reproduce the scalar
accumulation order, and `make test-cpu-quant` checks them against scalar block
arithmetic.

V4.1 Flash Q2 runs its routed experts with the arithmetic of the CUDA path:
the expert input and the SwiGLU mids are quantized to Q8_K with the formula
of `q8_K_quantize_kernel`, gate, up and down are integer IQ2_XXS and Q2_K
dots, the router weight is applied before down and the six down rows are
summed in F32 in slot order. Its logits are therefore not bitwise those of
Metal, which keeps F32 activations in the expert mat-vecs, nor of CUDA
(reduction order). Everything else in V4.1, and the Q4_K experts, keeps the
scalar reference arithmetic.

## Tests

```sh
make test-deepseek41-cpu      # V4.1 math, attention, payload, session and x86 kernel oracles
make test-cpu-quant           # V4 kernels against scalar block arithmetic, staged dispatches against the plain path
make test-deepseek41-cpu-live DS4_TEST_MODEL=/path/to/DeepSeek-V4.1-Flash-Q2.gguf
```

The live test loads the model and checks prefix reuse and a saved checkpoint
against a fresh prefill. For a change to the kernels, compare `ds4-bench --cpu
--teacher-forced-decode --dump-frontier-logits-dir DIR --dump-decode-logits-dir
DIR` dumps before and after: the V4 dumps must stay byte-identical.

## Measured

Dual Xeon Gold 6230 (2 x 20 cores, DDR4-2400, 512 GiB), 40 threads,
`--warm-weights`, warm page cache, `speed-bench/promessi_sposi.txt`, greedy
decode with `ds4-bench --cpu`; the host ran other services during the runs,
so treat the numbers as indicative.

| Model, protocol | Tokens/s |
| --- | ---: |
| V4 Flash Q2, 2048-token prefill | 106 |
| V4 Flash Q2, 128 tokens generated after 2048 | 9.6 |
| V4 Flash Q2, decode at a short context | 10.2-10.6 |
| V4.1 Flash Q2, 512-token prefill | 50 |
| V4.1 Flash Q2, decode at a short context | 9 |

The scalar CPU code these kernels replaced measured 1.14 tokens/s prefill and
0.66 tokens/s generation on the V4 protocol above, 1.9 tokens/s decode at a
short context, and 0.41 tokens/s decode for V4.1 Flash Q2, on the same
machine at 32 threads. The gap left to the memory bandwidth is small: a V4
decode token reads 9.5 GB of weights, about 76 ms at the 125 GB/s one process
gets from this box, against 93-96 ms measured on a quiet host.
