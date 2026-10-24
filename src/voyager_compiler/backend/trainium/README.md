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
2. `translate(model, out_dir, TRN2, immediates(gm, model))` writes
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

## Design rules

Follow these when extending this backend or adding another one.

1. **The translator only translates.** It reads one IR operation and emits
   its NKI instruction(s). It splits an operation only by fixed hardware
   limits that apply to every op alike: 128 rows per instruction, 512
   matmul output columns, 128-deep contraction chunks, and loads of 1 MiB
   or more in 256 KiB pieces. It does not fuse across operations, rewrite,
   pattern-match or plan; anything a kernel does is stated in the IR.
2. **PSUM is in the IR.** It is the IR memory level
   `MEMORY_LEVEL_LOCAL_BUFFER` (space tag `"Local"`). The builders allocate
   the PSUM buffers and state which one each matmul writes, the copies from
   PSUM to SBUF, a split-K sum kept in PSUM and attention scores kept
   there. NKI places on-chip buffers itself, so memory planning gives a
   PSUM buffer no address.
3. **A kernel that computes differently gets its own builder.** When a
   kernel's computation differs from the shared builder's in more than
   details, the backend has its own builder, registered ahead of
   `SHARED_BUILDERS` through `Backend.builders`. A kernel with the shared
   schedule subclasses the shared builder and overrides only its steps.
   Shared code changes only where a hook is needed, and Voyager's
   `model.txt` stays the same.
4. **Two levels of fusion, one representation.** General fusion runs in
   `transform()` on the model's own ops (`fuse_operator`, e.g. a GEMM and
   its dequantize tail). Instruction-level fusion groups the ops a backend
   builder creates into single NKI instructions (`fuse_patterns`); it runs
   inside each builder on the subgraph it just built, before bufferization
   splices it in, so the groups still get dtype stamping, renaming and
   layer tags. Both match a pattern the same way: a sequence of
   `OpMatcher`, one per op along a producer-consumer chain, each with an
   optional check of that op's operands. An instruction-level pattern also
   has a name, the NKI instruction form: the fused node records it,
   `model.txt` carries it in the fused op's `pattern` field, and the
   translator dispatches on it and raises on an unknown one. Builders emit
   plain aten / `quantized_ops` ops and do not decide the grouping.
   Voyager's patterns have no name, so its `model.txt` is unchanged.
5. **A custom op library is for whole-kernel swaps only**, the way
   `aten.sdpa` stands for an entire attention that a dedicated kernel
   implements. Do not define a custom op for a single instruction.
6. **Operand layouts are an argument, not an op variant.** A GEMM op takes
   one `layout` string, `"<activation>,<weight>-><output>"`, in the
   `--gemm_layout` letters (`x` the rows M, `c` the contraction K, `k` the
   output columns N): `"cx,ck->xk"`, or `"cx,ck->kx"` for an output stored
   `[N, M]`. Likewise a GEMM writing PSUM says so by its `out_dtype`
   argument (float32): its product is summed and returned in it.
7. **The op class's fit decides every buffer.** Each kernel's buffers at
   every memory level -- its SBUF tiles, its scratch and its PSUM -- are
   decided once, by the fit of the op's class in the runtime model, and
   returned in the placement (a backend may subclass `Placement`, as
   `TrainiumPlacement` does). The search stores the winner on the node
   (`meta["tile_result"]`), and the builders only read it and allocate
   what it lists. Nothing recomputes a size, and the translator creates
   no buffer of its own.
8. **Backend logic lives in backend code.** A shared option or hook that
   exists only to carry one backend's behaviour is that backend hardcoded
   in shared code. Shared code changes only by small, backend-neutral
   extensions.
9. **Trainium-only paths live in the Trainium subclass.** Everything
   K-major in flash attention -- the query and key stored
   `[head_dim, tokens]`, their tiles' addressing, a one-token query's
   transposing load, a transposed output -- is in `TrainiumFlashAttention`.
   The shared FA3 builder keeps Voyager's schedule, the improvements every
   backend takes and the hooks the subclass overrides.
10. **A transposing load is the operand's load functions.** The shared
    scheduler keeps all its loading logic; an operand's spec may carry
    `load_buffers` (called once before the loop) and `load` (called in
    place of `async_copy`). Trainium's GEMM builder sets both for an
    operand it transposes on TensorE (`load_transposed`); Voyager sets
    none.

## What is shared and what is custom

| Stage | Trainium |
|---|---|
| Export, PT2E quantization, `transform()` passes | shared as is; `LAYOUT` stores every GEMM weight contraction-major (`ck`, [in, out]), the activations between matrix-matrix GEMMs contraction-major too (`gemm_layout="cx,ck->xk"`), and keeps feature maps NCHW |
| Tile search (interstellar for GEMMs, the shared grid search for vector ops and GEMVs) | shared; it asks the target's runtime model (`tiling/runtime_model.py`) to prepare each op, then whether each candidate fits and what it costs |
| Runtime model | custom: `TrainiumModel` in `model.py`, preparing per op a `TrainiumGemm` or `TrainiumGemv` (`gemm.py`), a `TrainiumAttention` (`attention.py`) or a `TrainiumVector` or `TrainiumPool` (`vector.py`) from the graph facts in `tiling/lowering.py` |
| Bufferization, memory planning, `model.txt` | shared, but a dense GEMM (`linear`, `matmul`, their MX forms, a decode GEMV) lowers by Trainium's own builder, `gemm.py`, and a K-major flash attention by `attention.py` (`TrainiumFlashAttention`, a subclass of the shared FA3 pipeline), both registered ahead of the shared ones (`TrainiumConfig.backend`) |
| Translation to NKI | custom: `translate.py` |
| Simulation, trace, device runs | custom: `runtime/` |

## Files

One file per op, holding its op class (fit and cost) and its builder:

- `gemm.py`: `TrainiumGemm` and `TrainiumGemv`, and the GEMM builder
  (`build_gemm`).
- `attention.py`: `TrainiumAttention`, and the flash-attention builder
  (`TrainiumFlashAttention`).
- `vector.py`: `TrainiumVector` and `TrainiumPool`, which the shared
  builders lower.
- `utils.py`: what the op files share: `TrainiumPlacement`, the SBUF and
  PSUM sizes of placed buffers, the scratch regions the translator computes
  in, and `TrainiumGrid`, the fit of a vector op and a GEMV.
- `model.py`: `TrainiumModel`, which hands each op to its op class and
  prices what they share (an engine instruction, a DMA, an MX tile, a
  sweep). It imports the op files; they import only `utils.py`.
- `patterns.py`: the instruction patterns; `config.py`: `TrainiumConfig`,
  `TRN2` / `TRN3` and the builders' registration; `translate.py`: the
  translator to NKI; `runtime/`: simulation, tracing and device runs.

## How the IR becomes NKI

- **Control** (loops, conditions, index arithmetic) runs at translation
  time. A kernel is the straight-line sequence of transfers and compute
  steps the schedule makes. Semaphores and waits are dropped: NKI orders
  instructions by their data dependences.
- **SBUF tiles** have their rows (all dims but the last, flattened) on the
  partitions and the last dim free. A one-row tile read as an elementwise
  operand is replicated across the partitions when loaded. A tile one
  access pattern walks moves in one `dma_copy`, except a load of 1 MiB or
  more: that moves in pieces of 256 KiB or more of whole 128-row blocks,
  the last piece whatever blocks remain, so a GEMM's first matmuls start
  on the first piece of a weight tile and the next tile's pieces land as
  the current one's blocks free up. A tile no single pattern walks moves
  in one `dma_copy` per 128 rows, split further until a pattern walks each
  part. `transform` runs with `tensor_layout="nchw"`, so a pool's H lands
  on the partitions while the GEMM weights stay `"ck"`.
- **A 2-D max pool** loads each output row's kH input rows side by side
  onto that row's partition, at most 128 rows per tile, and runs kH-1
  vertical and kW-1 horizontal maxes on VectorE.
- **Access patterns** come from the elements' offsets: each instruction's
  operands are mapped element by element and factored into one shared
  nesting (`fit`). Views and relayouts fused into a kernel need no special
  case.
- **A GEMM step** is written out in the IR by Trainium's GEMM builder
  (`gemm.py`), on top of the shared plan and synchronous schedule. Its
  output tile is cut into slices of at most 128 rows and 512 columns, each
  one GEMM into a PSUM box, summed in float32 (`out_dtype`). With the
  reduction in one step the slices' GEMMs write the 4 rotating
  `[128, 512]` float32 accumulators in turn,
  each slice evicted at once: the fused tail run on it, or a copy, writes
  the output tile from PSUM. A reduction split over steps keeps the whole
  output tile's sum in PSUM, a bank per slice: the first step's GEMMs write
  it, the later ones add to it (`matmul_accumulate`), and the last step
  evicts each slice at once -- or every slice after its GEMMs, when the
  output tile is the sum itself. A row-major activation is transposed on
  TensorE per 128 rows, its 128 x 128 blocks side by side into a PSUM
  transposition tile of half a bank, each tile copied out whole into an
  SBUF tile the GEMMs read contraction-major. A tail ending in a transpose
  of the last two dims evicts the slices into a staging tile, then
  transposes the finished tile on TensorE into the output. The
  instruction patterns then fuse the builder's ops.
- **Buffers** are decided once, by each op class's fit, and
  listed in its `TrainiumPlacement`: `buffers`, the SBUF and PSUM tiles a
  Trainium builder allocates beside its streams (`gemm_buffers` for a GEMM
  or GEMV, `TrainiumAttention.buffers` for a flash attention), and
  `scratch`, the regions the translator computes an op group's
  intermediates in (`scratch_regions`: a chain's float32 pairs, a
  `quantize_mx`'s rearranged input and transposes, a K-major RMS norm's
  ones matrix and sums, a norm's statistics), which the builders attach to
  the op with `with_scratch`. An op built whole without a search takes its
  scratch from its op class (`OpModel.scratch`). A kernel's PSUM must fit
  the core's `psum_banks` x `psum_bank_bytes` (`fits_psum`; NKI reserves
  no bank for a kernel that never asks it to lower a `dma_transpose`), and
  the fit rejects a tiling whose buffers do not fit PSUM or SBUF. No SBUF
  is reserved (`scratchpad_offset` is 0): the translator declares the IR's
  boxes and nothing else.
- **PSUM boxes**: a box the IR places at `MEMORY_LEVEL_LOCAL_BUFFER` is
  declared in PSUM (`nl.psum`), laid out as an SBUF tile is but with an
  8-bit element two bytes apart. The ops writing it, and the fused groups
  that are one instruction, are in "Fused patterns" below.
- **A GEMM prologue**, the ops fused ahead of a row-major GEMM that
  compute its activation (an add, then an RMS norm), runs on each 128-row
  chunk into an SBUF tile before the transposes, a row op over whole
  rows, so `TrainiumModel` keeps the contraction whole. It reruns for
  every N tile, which the model prices.
- **A row-wise op** (layer norm, RMS norm, softmax) runs 128 rows at a
  time, one row per partition. A norm takes its rows' moments with
  `bn_stats` / `bn_aggr`; an RMS norm's mean square is `mean² + variance`,
  and one pass then scales each row by its rstd and the weight. The ops
  fused before it write its rows to the destination first, and it then
  runs there in place. A GEMM tail ending in one evicts each 128-row
  chunk's slices into the output tile, then the builder runs it on the
  chunk's rows, so `TrainiumModel` keeps an output tile's rows whole.
- **A K-major GEMM** (`linear` or `matmul` of layout `"cx,ck->xk"`) reads
  its activation `[..., K, M]` as stored, the contraction already on the
  partitions, so it transposes nothing: the activation is the stationary
  operand, or, when the GEMM writes its output transposed (`[..., N, M]`,
  layout `"cx,ck->kx"`), the moving one and the weight the stationary, and
  the slices run over 128 of N and 512 of M.
- **A K-major RMS norm** (`rms_norm_kmajor`) normalizes over the
  partitions: per 512 columns, the 128-row blocks are squared on ScalarE,
  8 blocks an instruction, and a ones-matmul on TensorE sums the squares
  over the blocks and the partitions into PSUM, landing the sum on every
  partition; one VectorE pass per block then scales by the rstd and the
  `[H, 1]` weight.
- **RoPE's rotate-half** is a K-major `matmul` by the `[D, D]` signed
  permutation constant, an ordinary K-major GEMM whose tail (`* sin`, `+ x *
  cos`) reads each slice from PSUM.
- **A flash attention** (`sdpa_kmajor`) lowers by Trainium's own builder,
  `attention.py`: `TrainiumFlashAttention`, a subclass of the shared FA3
  pipeline (`bufferize/attention_v3.py`) that keeps its walk, DMA streams
  and semaphores and states each step's instructions in the IR, per block
  of 128 query rows, written out unrolled. Everything K-major -- the query
  and key stored `[head_dim, tokens]`, their tiles' addressing, a
  one-token query's load and a transposed output -- is in
  `TrainiumFlashAttention`; the shared builder lowers only Voyager's
  attentions. QKᵀ writes the scores into PSUM in float32, a bank at a
  time; on a step that adds a mask (under `is_causal`, a pair crossing the
  diagonal), a matmul through an identity of `1 / scale`, built once on
  chip (`scaled_identity`), adds it onto them there (`matmul_accumulate`).
  Per block, one VectorE `tensor_scalar_reduce` per bank evicts the scores
  scaled with their row max, a `tensor_scalar` makes the bias `-max(m,
  max)`, and one ScalarE `activation` takes the exponentials in place with
  their row sum. Alpha, `m` and `l` then run over every block's float32
  columns at once. P is transposed on TensorE into PSUM and copied into an
  SBUF Pᵀ, which P·V reads; P·V's result stays in PSUM for the rescale `o
  = alpha·(o + pv)` to read. A scaling by alpha or `l` runs on VectorE,
  since ScalarE runs the exponentials. Scores wider than a PSUM bank are
  evicted a bank at a time, each bank's row max into a column of its own,
  then one VectorE pass takes the max of those columns.
- **A decode step's flash attention** is the same kernel over one query
  token. The layout pass stores the key cache `[head_dim, positions]`, so
  QKᵀ reads it as stored, and the GQA fold puts the query heads of a KV
  head (4 in Llama-3.1-8B) on one query tile, loaded row-major and
  transposed on TensorE through PSUM once per KV head, and its output
  transposed back the same way, all stated in the IR. A 1152-slot cache is
  one key block, so each KV head is one step, and the mask, broadcast over
  the heads, is one tile loaded once.
- **A transposing load** (an operand a GEMM reads transposed from how it
  is stored) lands as stored into staging slots by DMA, then one commit
  transposes it on TensorE through a PSUM tile into its slot
  (`load_transposed`, the operand spec's `load` and `load_buffers`): the
  blocks of one block row of the result go side by side into a PSUM tile
  of up to a bank, as the kernel's other PSUM tiles leave room for, and
  are copied out together. TensorE reads a block from partition 0 only,
  so a load of several leading elements whose rows do not fill whole
  128-row runs (per-head rows under 128) is transposed by the DMA instead,
  element by element.
- **A layer norm** normalizes, scales and shifts in its output tile and
  keeps its statistics in the scratch regions its op class lists.
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
  references packed this way.  A `linear_mx` slice runs one
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

## Fused patterns

Fusion runs at two levels, with one representation: a `Pattern`
(`voyager_compiler`), the `OpMatcher` of each op along a producer-consumer
chain, each optionally checking that op's operands, and an optional name.

- **General fusion** runs in `transform()` on the model's own ops
  (`fuse_operator`), by the unnamed patterns it is given: a GEMM or a norm
  and the ops after it, any of which a group may skip.
- **Instruction-level fusion** runs inside each Trainium builder on the
  subgraph it just built: `fuse_patterns(gm, INSTRUCTION_PATTERNS)`, in that
  subgraph and the loop bodies and branches it calls. Each pattern in
  `patterns.py` is named for the NKI instruction it is. A chain matches only
  with an op for every matcher, and where two patterns match one op the
  earlier takes it. A group's results are its values read outside it, in
  program order: its last op's, and the tile a reduction marked
  `keeps_input` reads, which the instruction stores as it reduces it (a
  chain whose tile is read only by the reduction stays unfused). The group
  runs where its last op stood, so a builder reduces a tile before storing
  it.

The fused node records the pattern's name (`meta['pattern']`), `model.txt`
carries it as the fused op's `pattern`, and the translator emits the group
as that instruction, one per 128 rows (`PATTERN_HANDLERS`), raising on a
name it does not know. A custom op library is only for swapping a whole
pattern for a dedicated kernel, the way `aten.sdpa` stands for a whole
attention, never for a single instruction.

| pattern | ops in the group | NKI instruction | why it is one instruction |
|---|---|---|---|
| `tensor_scalar` | an ALU op (`add`, `sub`, `mul`, `maximum`) of a tile and a column of one value per row, then one of its result and a constant | `tensor_scalar` with two operators, on VectorE | the engine applies two operators to each element in one pass, e.g. the softmax bias `-max(old max, new max)` as `maximum` then `mul` by -1 |
| `tensor_scalar_reduce` | an ALU op of a tile and a constant, a cast (`to`) when it is stored in another type, then `amax` or `sum` over its last dim, kept | `tensor_scalar_reduce` | the engine reduces each row of its result as stored in the same pass, e.g. scores copied out of PSUM scaled, with their row max |
| `activation_reduce` | `add` of a tile and a column, `exp`, then `sum` over its last dim, kept | `activation` with `reduce_op=nl.add` | ScalarE adds the bias before the function and sums each row of the result in the same pass, e.g. a softmax's exponentials with their row sums |
| `scaled_identity` | `eye`, then `mul` by a constant | `memset` of the constant and `affine_select` keeping the diagonal | the identity a mask is added through, scaled by `1 / scale`, is made on chip once |
| `matmul_accumulate` | a K-major GEMM (a `matmul`, or a `linear` without a bias, of activation layout `"cx"`) or an MX one (a `linear_mx` without a bias), then `add` of its product and the PSUM partial sum the group writes | `nc_matmul` (`nc_matmul_mx`) with `accumulate=True` | TensorE adds its product into the PSUM tile it writes, e.g. a split reduction's later steps, or a mask added onto scores by a matmul through an identity |

To add an instruction: append a named `Pattern` to `INSTRUCTION_PATTERNS`,
one `OpMatcher` per op with a predicate on that op's operands (the ones in
`patterns.py`, or a new small one), marking a reduction the instruction
makes of the tile it stores `keeps_input=True`; a form whose op is optional
is two patterns of one name. Then add the `Kernel` method emitting it to
`PATTERN_HANDLERS` in `translate.py` under the name, and a test in
`test/unit/test_trainium.py` that `fuse_patterns` forms the group from plain ops
and that it translates to the instruction.

A lone op writing a PSUM box is the instruction that writes PSUM: an
`aten.transpose` one `nc_transpose` per 128 x 128 block, a K-major GEMM one
`nc_matmul` per 128 rows, 512 columns and 128-deep contraction chunk, a
`linear_mx` one `nc_matmul_mx` per 128 rows, 512 columns and 512-deep
chunk. An `aten.eye` is a `memset` of ones and an `affine_select` keeping
the diagonal, per 128 rows. An op reading a PSUM box of whole 128-row
blocks runs on all its blocks at once, side by side in their bank.

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

A GEMM writing PSUM sums in float32 in the reference too (`out_dtype`),
so a split-K GEMM rounds once, as the kernel does.  An FP8 result may
round one step the other way.

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
unpadded NCHW max_pool2d, and a fused attention other than a K-major flash
attention (`sdpa_mx`, a split KV cache's, is not lowered).
