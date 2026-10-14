# GPU backend

Runs a Voyager-quantized graph on a GPU with real low-precision GEMMs
instead of fake-quant: FP8 linears and attention matmuls, and NVFP4
linears. The Voyager backend is untouched and imports nothing from here.

## How a backend is chosen

| Backend | Entry points |
|---|---|
| Voyager | `voyager_compiler.transform()` / `.compile()` (unchanged) |
| GPU | `voyager_compiler.backend.gpu.lower()` on the graph `convert_pt2e` produces |

`benchmarks/gpu/run_lm.py` drives the GPU backend for Llama.
`test/test_codegen.py` is for Voyager compilation only.

## What is shared and what is custom

| Stage | GPU | Note |
|---|---|---|
| Model loading, export and quantization (`llama.load_model`, `llama.quantize_model` from `test/utils/models/llama.py`) | shared as is | run under `with torch.device(gpu):`, so the inputs it builds land on the GPU |
| Quantization specs, quantizer, `prepare_pt2e`, `convert_pt2e` | shared as is | |
| `run_shared_passes`, `transform()`, `compile()` | not used | cuBLAS tiles each GEMM; PyTorch's allocator places memory |
| FP8 GEMMs (`fp8.py`, `lower()`) | custom | |
| NVFP4 cast and GEMM (`nvfp4.py`, `lower()`) | custom | the cast repeats Voyager's arithmetic; the GEMM is torch's block-scaled `F.scaled_mm` |
| Fusion | `torch.compile` | each activation cast fuses into the elementwise op before it |

## What `lower()` does

`convert_pt2e` leaves an FP8 linear as

```
quantize(x, s_a) -> linear(., W_q) -> dequantize(., s_a * s_w)
```

and an FP8 matmul of two activations as

```
quantize(a, s_a), quantize(b, s_b) -> matmul -> dequantize(., s_a * s_b)
```

with the codes held as bfloat16 values on the FP8 grid; a direct cast
(`fp8_e4m3` with no scheme) has unit scales and no dequantize. `lower()`:

1. casts each `W_q` to FP8 once and drops the bfloat16 copy;
2. adds one cast per activation `quantize`,
   `(x / s_a).clamp(-max, max).to(fp8)`, read by every GEMM it fed; the
   `quantize` stays only for a non-GEMM reader;
3. replaces each linear and its dequantize with one `torch._scaled_mm`
   (`fp8_linear`): a scalar scale per tensor, row-wise scales per output
   channel, the bias multiplied back out of the GEMM's scale;
4. replaces each matmul and its dequantize with one
   `torch._scaled_grouped_mm` over the flattened batch dims (`fp8_bmm`),
   e.g. eager attention's QK^T and PV, one GEMM per batch and head.

An NVFP4 linear (`fp4_e2m1` in blocks of 16 along K, `fp8_e4m3` block
scales, one global scale per tensor) is left as

```
t = clamp_min(amax(|x|) / (6 * 448), tiny)
s, q = quantize_mx(x, ..., global_scale=t)
linear_mx(q, W_q, input_scale=s, weight_scale=S_w, alpha=t_w * t)
```

and `lower()`:

1. packs each `W_q` and `S_w` once (`pack_nvfp4`) and drops the bfloat16
   copies;
2. replaces each activation `quantize_mx`, with the amax chain before it,
   by one `quantize_nvfp4`, read by every linear it fed; `alpha` reads the
   global scale the cast returns;
3. replaces each `linear_mx` with one `nvfp4_linear`, a `voyager_gpu` op
   over uint8 buffers, so Inductor never handles the FP4 dtype.

| NVFP4 operand | Layout `F.scaled_mm` reads |
|---|---|
| codes | two `fp4_e2m1` codes per byte along K, the even element in the low nibble (`torch.float4_e2m1fn_x2`): the activation `[M, K/2]` row-major, the weight `[N, K/2]` read column-major |
| block scales | `fp8_e4m3` bytes in cuBLAS's 128x4 tiles (`to_blocked`, `SWIZZLE_32_4_4`), rows zero-padded to 128 and K/16 to 4 |
| 16x16 weight blocks | codes as they are; each block's scale repeated over its 16 rows, since the GEMM reads one scale per 16 elements of a row |
| alpha | Voyager's bfloat16 `t_w * t`, copied to float32 in the graph, as A's second-level (`TensorWise`) scale and 1, one buffer every GEMM shares (`nvfp4_one`), as B's: applied to the float32 product inside the GEMM, the bias added after it, as `linear_mx` does |

Everything else runs as written, fake-quant included. `lower()` returns,
per GEMM node, `"fp8"`, `"nvfp4"` or why it was left alone:

| Left as is | Why |
|---|---|
| conv; MX ops but an NVFP4 `linear_mx`, e.g. an NVFP4 `matmul_mx` | only `aten.linear`, `aten.matmul` and NVFP4 `linear_mx` are lowered; torch has no batched block-scaled GEMM |
| matmul operand that is not a `quantize`, e.g. the rotary matmul (unquantized) or decode's main value cache, which `convert_pt2e` quantized in place | no FP8 activation to cast |
| K or N not a multiple of 16 | the FP8 GEMMs' alignment |
| NVFP4 N not a multiple of 16 or K of 32 | the NVFP4 GEMM's alignment, K counted in bytes |
| E5M2 x E5M2 linear; any E5M2 matmul operand | no such FP8 GEMM on CUDA; the batched one takes E4M3 only |
| per-channel, block, zero-point or codebook activation quantize | not one FP8 scale |
| NVFP4 activation without a global scale, not in blocks of 16 along K, or not bfloat16 | not the recipe `quantize_nvfp4` repeats |
| a GEMM the GPU cannot run | see below |

| GEMM | Runs on |
|---|---|
| FP8 linear (`fp8_linear`) | CUDA compute capability 8.9+; MI355X (gfx950), not MI300 (gfx94x), whose FP8 is E4M3FNUZ, a different grid |
| FP8 matmul (`fp8_bmm`) | the same, on CUDA 9.x and 10.x only (`torch._scaled_grouped_mm`): on sm_120 attention matmuls stay fake-quant |
| NVFP4 linear (`nvfp4_linear`) | CUDA compute capability 10.x or 12.x, CUDA 12.8+ |
| every GEMM on CUDA | a torch build with code the GPU runs in `torch.cuda.get_arch_list()`: `sm_XZ` or `compute_XZ` with Z no higher than the GPU's minor, so sm_89 runs the cu126 build's sm_86; that build stops at sm_90 |

## Numerics

- Voyager's `fp8_e4m3` / `fp8_e5m2` are torch's `float8_e4m3fn` /
  `float8_e5m2`: the cast picks Voyager's code for every bfloat16 value
  (`test_cast_rounds_as_voyager`).
- A GEMM is not bit-exact with the fake-quant graph: that graph rounds the
  GEMM output to bfloat16 and then to the `--bias` dtype before scaling,
  while the FP8 GEMM scales the float32 sum and rounds once.
  `check_sites()` measures each site on the converted graph's own
  activations: the error is about an ulp of the largest output.
- End to end the logits differ more, by a few percent, since a 1-ulp
  difference flips FP8 codes in later layers.
- Under `torch.compile`, Inductor rounds a fused chain's float32
  intermediates once rather than at each bfloat16 op, which can flip FP8
  codes too. With `eager_numerics.division_rounding` and
  `emulate_precision_casts` the compiled graph matches eager exactly.
- `quantize_nvfp4` repeats `quantize_mx`'s bfloat16 arithmetic op for op
  (`amax / 6`, then `/ t`, each rounded to bfloat16), so it picks Voyager's
  codes and block scales; casts that keep the scale in float32 (torchao, TE)
  differ in about 1% of codes. Its stochastic rounding draws Voyager's
  Philox bits, indexed by element in the tensor `quantize_mx` sees, and its
  random Hadamard transform is Voyager's `_rotate`. On the CPU: 0
  mismatches on random and edge tensors, with and without either.
- Unpacked NVFP4 operands are exact in bfloat16, so the emulator
  (`emulate_nvfp4_linear`: unpack, then `linear_mx`) gives the converted
  graph's output bit for bit. Like the real GEMM, `linear_mx` scales the
  float32 sum by alpha, adds the bias and rounds once; they differ only in
  the order the sum is accumulated.

## How to run

Pick a GPU with no other job on it.

```
CUDA_VISIBLE_DEVICES=<gpu> python -m pytest test/test_gpu_fp8.py

# NVFP4: the cast, packing, emulated lowering and device gates run on a
# CPU; the real GEMM's tests need sm_100 or sm_120
CUDA_VISIBLE_DEVICES=<gpu> python -m pytest test/test_gpu_nvfp4.py

# Llama-3.1-8B in BF16, per-tensor FP8 and NVFP4; --schemes also takes
# fp8_channel, and --implementation te runs the same schemes on TE
python benchmarks/gpu/run_lm.py --out_dir <dir> \
    --schemes bf16 fp8_tensor nvfp4 [--stages prefill decode ppl] \
    [--device cuda:<gpu>]

# The training lowering: emulated on a CPU, the real GEMMs on a GPU
CUDA_VISIBLE_DEVICES=<gpu> python -m pytest test/test_gpu_training.py
```

`run_lm.py` reports per scheme and stage: GEMMs lowered, codes that differ
from the fake-quant graph, the worst site error, the logits' error against
the fake-quant graph and against the HF model in BF16, latency under
`torch.compile` without and with a CUDA graph (`prefill`, `decode`), and
wikitext-2 perplexity of the fake-quant and the lowered graph (`ppl`).

## Training

`training.lower_training(program)` lowers a training step's whole program,
as `gradient_program` writes it, in place: every quantized GEMM -- forward,
input gradient and weight gradient -- becomes `fp8_linear` or
`nvfp4_linear`, each quantized tensor cast once. The FP8 GEMM takes both
operands K-major, so a backward GEMM reads a transposed copy of the FP8
codes; a weight in 16x16 NVFP4 blocks is cast once
(`quantize_nvfp4_square`) and packed both ways, so the input gradient reads
the forward's codes. Stochastic rounding keeps the program's seeds, and the
random Hadamard transform's rotations stay in front of the casts.
`test/test_gpu_training.py` checks the lowered program against the
fake-quant one; `benchmarks/gpu/train_lm.py` times it against TE.

## Status

- `test/test_gpu_fp8.py`: 15 pass on an H100 (2026-10-05).
- `run_lm.py` on AMD-Llama-135m cut to 2 layers, per-tensor FP8, eager
  attention: prefill lowers all 15 linears and 4 attention matmuls, decode
  the 15 linears and the 2 main-cache QK^T matmuls; FP8 codes match the
  fake-quant graph's, worst site error 7.6e-3.
- `test/test_gpu_nvfp4.py` on the CPU (2026-10-13): every test that does
  not run the real GEMM passes. A two-layer Llama with the `nvfp4`
  config, its weights in 1x16 or 16x16 blocks, lowers all 15 linears, and
  with the GEMM emulated gives the fake-quant graph's output exactly. Fake
  CUDA tensors pass torch's argument checks for the NVFP4 GEMM at
  Llama-3.1-8B's shapes, 1 and 1000 rows.
- On an RTX PRO 6000 (sm_120, 2026-10-13): the real FP8 and NVFP4 GEMMs
  pass their checks against the emulators, and Llama-3.1-8B prefill and
  decode lower every linear (NVFP4: 0 codes differing from the fake-quant
  graph's). Results: `~/claude/runpod/results`. That run predates the
  removal of the single-level NVFP4 GEMM and `linear_mx`'s float32 alpha.

## Open items

- Compiled on the CPU, the NVFP4 cast drops the bfloat16 rounding between
  `amax / 6` and `/ t` even with the options above: 80 of 512 block scales
  differ from eager on a test tensor. Unchecked on a GPU; checks of codes
  and outputs run eager.
- NVFP4 attention matmuls (`matmul_mx`) stay fake-quant.

- `sdpa` attention runs as written in bfloat16; FP8 attention through it
  would take FlashAttention-3's FP8 kernel.
- Decode's main PV matmul reads the value cache buffer, which
  `convert_pt2e` quantized in place, and stays fake-quant.
- A float bias dtype wider than bfloat16 (`fp32_e8m23`) fails when the bias
  fake-quant builds its lookup table on the CPU: the format's largest value
  overflows bfloat16.
- `convert_pt2e` takes the model's dtype from its first parameter, so a
  model with none cannot be converted.
