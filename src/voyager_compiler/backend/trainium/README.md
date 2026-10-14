# Trainium backend

Runs Voyager's own schedule on one AWS Trainium NeuronCore. Voyager compiles
the model as it would for its own accelerator: tile search, bufferization
and memory planning. Its `model.txt` is the hand-off, and this backend only
translates it into NKI, deciding nothing.

## How a model gets there

1. `transform()` / `compile()` with `config=TRN2` (or `TRN3`) and
   `**LAYOUT`. The tile search asks `TRN2.runtime_model(accumulate_fp32)`,
   a `TrainiumModel`, what fits SBUF and what each tile costs, instead of
   Voyager's own model.
2. `translate(model, out_dir, immediates(gm, model))` writes
   `kernels.py`, `program.py` and `manifest.json`.
3. `record(gm, args, manifest, out_dir/"tensors.pt")` saves every HBM
   box's value as Voyager's bufferized graph computes it: the reference.
4. In the `trn` env (`source ~/claude/trn/env.sh`):
   - `runtime/check.py <dir>` simulates each kernel on CPU against the
     reference (`--program` runs the whole step, `--neff --target trn2`
     compiles each distinct kernel for that target, under
     `<dir>/neff/<target>/`);
   - `runtime/trace.py <dir> --phase prefill` traces the step into one
     NEFF, with NxD Inference's flags and decoder-layer markers;
   - `runtime/run.py <dir>`, on a Trainium host, times it.

`benchmarks/trainium/compile_lm.py` does steps 1-3 for a causal LM and
writes `predicted.json`, the latency the tile search predicted for each
kernel.

## What is shared and what is custom

| Stage | Trainium |
|---|---|
| Export, PT2E quantization, `transform()` passes | shared as is; `LAYOUT` stores every GEMM weight contraction-major (`ck`, [in, out]) and keeps feature maps NCHW |
| Tile search (interstellar for GEMMs, the shared grid search for vector ops and GEMVs) | shared; it asks the target's runtime model (`tiling/runtime_model.py`) to prepare each op, then whether each candidate fits and what it costs |
| Runtime model | custom: `TrainiumModel` in `model.py`, preparing a `TrainiumGemm`, `TrainiumGemv` or `TrainiumVector` per op from the graph facts in `tiling/lowering.py` |
| Bufferization, memory planning, `model.txt` | shared as is |
| Translation to NKI | custom: `translate.py` |
| Simulation, trace, device runs | custom: `runtime/` |

## How the IR becomes NKI

- **Control** (loops, conditions, index arithmetic) runs at translation
  time. A kernel is the straight-line sequence of transfers and compute
  steps the schedule makes. Semaphores and waits are dropped: NKI orders
  instructions by their data dependences.
- **SBUF tiles** have their rows (all dims but the last, flattened) on the
  partitions and the last dim free. A one-row tile read as an elementwise
  operand is replicated across the partitions when loaded. `transform` runs
  with `tensor_layout="nchw"`, so a pool's H lands on the partitions while
  the GEMM weights stay `"ck"`.
- **A 2-D max pool** loads each output row's kH input rows side by side
  onto that row's partition, at most 128 rows per tile, and runs kH-1
  vertical and kW-1 horizontal maxes on VectorE.
- **Access patterns** come from the elements' offsets: each instruction's
  operands are mapped element by element and factored into one shared
  nesting (`fit`). Views and relayouts fused into a kernel need no special
  case.
- **A GEMM step** transposes its activation on TensorE, 128 x 128 at a
  time, multiplies it into PSUM one 512-column slice at a time, and evicts
  each slice through the ops fused after the GEMM.
- **A GEMM prologue**, the ops fused ahead of a GEMM that compute its
  activation (an add, then an RMS norm), runs on each 128-row chunk into
  an SBUF temporary before the transposes, every op in place there: a
  pointwise op in column runs, a row op over whole rows, so
  `TrainiumModel` keeps the contraction whole. It reruns for every N
  tile, which the model prices.
- **A row-wise op** (layer norm, RMS norm, softmax) runs 128 rows at a
  time, one row per partition. A norm takes its rows' moments with
  `bn_stats` / `bn_aggr`; an RMS norm's mean square is `mean² + variance`,
  and one pass then scales each row by its rstd and the weight. The ops
  fused before it write its rows to the destination first, and it then
  runs there in place. A GEMM tail ending in one evicts each 128-row
  chunk's slices into the output tile, then runs it on the chunk's rows,
  so `TrainiumModel` keeps an output tile's rows whole.
- **A transposing load** (attention's Kᵀ) lands as stored, then is
  transposed on TensorE.
- **A layer norm** normalizes, scales and shifts in its output tile and
  keeps its statistics in the translator's temporaries, so the IR declares
  no scratch region on Trainium.
- **Immediates** (constants the IR names but does not hold, such as the
  attention scale) are written into the kernel as literals.
- **DRAM-only ops** (embedding, slice, cat) run in the host program as
  torch ops.

- **MX (Trn3)** arrays -- the codes and E8M0 block scales a `linear_mx`
  multiplies -- are stored the way `nc_matmul_mx` reads them (`Mx` in
  `translate.py`): 4 consecutive contraction elements per 32-bit word, a
  chunk of 512 on the 128 partitions, each further chunk beside it, and
  each block's scale on the first 4 partitions of its quadrant.  In HBM
  that is `[128, K/512, X]` int32 codes and `[16, K/512, X]` uint8 scales,
  `X` the activation's rows or the weight's columns; `record` saves the
  references packed this way.  A `linear_mx` step runs one
  `nc_matmul_mx` per 512 of the contraction, with no transposes (a single
  row is padded to the even width the instruction needs).  An op ending
  in `quantize_mx` writes its result in BF16, transposes every 4th column
  onto the partitions on TensorE and quantizes there with
  `nisa.quantize_mx`.  The tile search keeps MX contractions in whole
  chunks of 512 (`TrainiumModel`).

- **Host statements** are everything outside a kernel: DRAM-only ATen ops
  (embedding, slice, cat, a KV-cache `index_copy_`) called by schema
  keyword, an in-place op as its functional twin rebinding what it
  mutates, scalar arithmetic on host values, a compute op writing an HBM
  window, and HBM allocations no kernel makes. The parameters the program
  writes (its state) are stored back so a trace can alias them.

## Verified (one decoder layer, 1024 tokens)

Kernels passing the CPU simulator against Voyager's reference, and
compiling for the target (distinct kernels): trn2 for BF16 and FP8, trn3
for MX (MXFP8 E4M3).

| Design point | Llama-3.1-8B simulator | Llama-3.1-8B compiled | Qwen3-4B simulator | Qwen3-4B compiled |
|---|---|---|---|---|
| BF16 prefill | 20/20 | 17/17 | 22/22 | 20/20 |
| BF16 decode | 37/37 | 30/30 | 39/39 | 33/33 |
| FP8 prefill | 20/20 | 20/20 | 22/22 | 22/22 |
| FP8 decode | 34/34 | 32/32 | 36/36 | 34/34 |
| MXFP8 prefill | 20/20 | 17/17 | 22/22 | 20/20 |
| MXFP8 decode | 34/34 | 29/29 | 36/36 | 32/32 |

Every step traces into one NEFF for its target.  All eight Trn2 steps run
on a Trn2 device and match Voyager's output (cosine 0.9993 to 0.99998,
the same predicted token).  The MX steps have only run in the simulator:
whole steps at cosine 0.99965 and 0.99959 (Llama and Qwen prefill),
0.99943 and 0.99836 (decode).  An MX output is checked as values, block
by block: an element may round one FP8 step the other way where its BF16
input differs by a rounding, and those flips add up over a step's four
quantizations, so a whole MX step passes at 0.998 (E4M3) rather than
0.999, and at 0.992 for E5M2, whose step is twice as coarse.  MXFP8 E5M2
passes the same checks for Llama prefill (whole step 0.99797).

FP8 is Trn2's own E4M3, the dtype `fp8_e4m3_ieee`: the IEEE kind, which
keeps its top exponent for inf and NaN and so tops out at 240 (Voyager's
`fp8_e4m3` is OCP's E4M3FN, up to 448).

The BF16 prefill programs run whole in the simulator at cosine 0.99993
(Llama: argmax and top-5 equal), and every design point's step traces into
one trn2 NEFF.  The 32-layer Llama prefill translates to 578 calls of the
same 17 kernels and traces into one trn2 NEFF in about two minutes: the
decoder-layer markers let the compiler build one layer for all 32.

A split-K BF16 GEMM differs from Voyager's reference by about 1%: both
round their partial sums to BF16, in a different order, and both are about
1% from an FP32 matmul.  An FP8 result may round one step the other way.

A value has one destination in `model.txt`: decode's rotary
`cat(freqs, freqs)` stores its angles with two clones, one per half
(`bufferization._dedup_regions` never merges two stored values, and
`emit` rejects a second store of one).

A reference through a reshape of a quantized box gives the view's
physical dtype (`bfloat16`) rather than the box's (`fp8_e4m3`), a
difference in the shared compiler; the translator takes each box's dtype
from its declaration.

The simulator is functional only and lenient: four hardware rules surfaced
only in the trn2 compile -- no `mod` or `divide` ALU op, per-row scalar
operands in FP32, no free dim crossing partitions, FP8 transposes at stride
2 -- so every translation is compiled too (`check.py --neff`).

An `fp8_e4m3_ieee` step crosses into XLA as torch's `float8_e4m3fn`,
which has the same bits up to 240: `trace.py` sets NKI's non-OCP mode and
NxD Inference's `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`.

A traced decode step is specialized to its example position: the host
program reads the position as a Python scalar, which a trace cannot, so
it takes the value the reference run recorded -- a decode is compiled
with references (without `--no-reference`).

## Not supported yet

Kernels that write an HBM box they did not allocate, transfers with a
non-zero pad, partial transfers, MXFP4, convolution, pooling other than an
unpadded NCHW max_pool2d, and fused attention.
