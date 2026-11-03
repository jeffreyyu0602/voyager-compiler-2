# Change log

What each commit changed in the compiler's output: the Voyager `model.txt`
files of the CI configs (`test/run_ci.py`), the runtime model's predicted
latency, the numerics, and measured speed. Newest commit first. Only commits
that changed something are listed. How entries are made: README.md,
"Development workflow".

## 8d0ffa7 → (2026-11-03)

From the CI run of 2026-11-03 (`test/run_ci.py`, baseline e5c49a0; run
folder `/sim/jeffreyy/soc-voyager-migration/sims/vcs/claude-scratch/weight-layout/ci/2026-11-03_03-27-38`).
The commits from fd6feef to 8d0ffa7 are not logged yet.

### fix: store the KIVI cache in place (2026-11-03)

- Not in the CI run above; traced on the `llama_cache_kivi` pair only.
- **model.txt, llama_cache_kivi:** a KIVI chunk's cache store writes into a
  pitched window of the cache, so the prefill and the decode's fold no
  longer `index_copy_` the quantized cache and its scales and zero points;
  the decode's per-token residual write stays a host copy. The layer list
  is unchanged.
- Numerics: the 8 cache, scale and zero-point buffers, both residuals and
  the logits are bit-identical.

### feat: store each weight once, in the model's layout (2026-11-03)

- **model.txt, 19 of 34 configs** (the 10 llama_prefill and llama_cache
  configs, bert, resnet18, resnet18_db, resnet50, vit and the 5
  mobilebert_encoder configs): a GEMM weight is stored in the model's
  layout (`[out, in]` by default) instead of a private transposed `_t` copy,
  with its scales likewise, and the GEMM kernels reading it load each tile
  transposed. Their allocs change shape and addresses move.
- Predicted latency, DRAM traffic and peak DRAM and scratchpad: unchanged in
  all 19. Numerics: all 34 configs keep their verdicts.
- The new `llama_cache_kivi` command (the Voyager `llama_prefill_mp` +
  `llama_decode_kivi` pair) is not in this run's comparison.

## 0a94c31 → fd6feef (2026-11-01, 6 commits)

From the CI run of 2026-11-01 (`test/run_ci.py`, baseline 0a94c31; run
folder `~/claude/ci/full/out/2026-11-01_12-20-25`). Predicted latency is the
runtime model's estimate (`--report`). The 17 resnet, mobilebert, mobilenet
and vit configs did not change. 18004f8 (4-bit training experiments),
b988cec (one-thread numeric check), 3780fc3 (driver moved to `scripts/`) and
f52d153 (the CI itself) changed no config.

### fd6feef fix: the latency estimate reads no 1-element constant (2026-11-01)

- **bert/MXNF4 is estimated again.** 3383941 exports the whole HF BERT model
  from `input_ids`, which brings HF's mask-building `index` into the graph
  with a 1-element int64 constant. A 1-element constant is passed by value
  and owns no DRAM, so `--report` raised in `_copy_traffic`, before the
  numeric check ran. The estimate now skips such a constant. bert/MXNF4
  passes its numeric check, and predicts 4,205,932 → 4,253,881 cycles
  (+1.14%; DRAM writes +3.8%), with the embeddings and the mask now part of
  the compiled graph.

### 3383941 feat: one quantized LM object whose prefill fills decode's KV cache (2026-11-01)

- **model.txt, the 5 llama_prefill configs:** the prefill writes a static KV
  cache.
  - New input `cache_position`, and new bf16 parameters `key_cache_0` and
    `value_cache_0` [1,8,1152,128] (1024 prompt + 128 generation slots),
    which two `index_copy_` kernels write (2 MB each). Attention still reads
    the fresh K and V, so QKᵀ is unchanged (67.1 MB of bf16 scores).
  - The causal mask (int1 [1,1,1024,1024]) and RoPE's cos/sin tables
    ([1,1,1024,128]) are no longer constants. New kernels build them from
    `cache_position`: `le`, `where` and a quantize for the mask; `matmul`,
    `cos`, `sin` and `mul` for RoPE.
  - K's and V's MX quantizes no longer fuse into their producers (K's RoPE
    `mul_6`, and `v_proj`): both are written in bf16 (2 MB each instead of
    0.5 MB) for the cache, and two separate kernels (`quantize_mx_default_9`,
    `_10`) quantize them for attention.
  - Kernels are renamed `model_layers_0_*` → `model_model_layers_0_*`,
    after the cache wrapper module.
- **model.txt, llama_prefill_mp, _spmm and _spmm_db** (the configs given a
  `--qconfig`) **and llama_decode_kivi:** softmax and RMSNorm inputs are no
  longer quantized to fp8. The driver's fp8 annotation of them is gone, so
  the attention scores stay bf16: QKᵀ writes 67.1 MB instead of 33.5 MB.
  The commit message does not mention it.
- **model.txt, llama_decode and _db (MXNF4):** the key cache is no longer
  split into a quantized part and a bf16 residual. The driver now splits a
  cache only when given `--residual_length`, and `transform()` then splits
  only a cache whose MX blocks run along positions, which is V's.
  - The new key is MX-quantized as it is written: `mul_6` writes 528 B
    instead of 2048 B.
  - Gone: the key residual, its score GEMM (`matmul_3`), and the two kernels
    that joined the score halves (`index_copy`, `mul_8`, 72 KB each). With
    one MX consumer left, Q's quantize fuses into RoPE
    (`quantize_mx_default_1` removed).
  - V's residual holds 64 positions instead of 128.
- **model.txt, llama_decode/BF16:** unchanged. Its outputs changed (below).
- **model.txt, the 3 llama_train configs:** training goes through
  `QuantizedModel`.
  - Parameters are renamed `model_model_*` → `model_*`, and
    `model_lm_head_weight` → `lm_head_weight`. That is the whole change to
    the update program: its 61 kernels differ by name only.
  - In the gradient program, two host `cat` ops that joined K and V onto an
    empty cache are gone.
  - In BF16 only, RoPE's frequency product (64 × 1024, fp32) is a `matmul`
    with a padded frequency constant and 64×512 tiles, instead of a `bmm`.
- **Predicted latency:**

  | Config | total_latency (cycles) | DRAM writes |
  |---|---|---|
  | llama_prefill, _db | 75,465,051 → 76,085,683 (+0.82%) | +7.2% |
  | llama_prefill_mp | 73,860,781 → 76,267,435 (+3.26%) | +23.6% |
  | llama_prefill_spmm, _spmm_db | 78,977,883 → 81,366,840 (+3.02%) | +19.5% |
  | llama_decode, _db (MXNF4) | 11,980,600 → 11,944,340 (−0.30%) | −22.3% |
  | llama_decode_kivi | 11,979,298 → 11,980,246 (+0.01%) | +4.4% |
  | llama_decode/BF16 | unchanged | unchanged |

  - Prefill, _db: the run-time mask (0.21M cycles), the two cache writes
    (0.13M), the separate K and V quantizes (0.10M) and RoPE's new kernels.
  - _mp and _spmm: on top of that, bf16 instead of fp8 scores cost QKᵀ
    0.95M and softmax 1.05M cycles, and the 3 RMSNorms 0.2M; the 2 fp8
    quantize kernels removed save 0.37M.
  - Decode MXNF4: the key-residual GEMM (21,760 cycles) is gone, and the
    V-residual GEMM drops from 21,760 to 13,296 cycles.
  - Training configs print no predicted latency.
- **New configs:** each compiles a prefill and the decode that reads the
  cache it writes, sharing DRAM addresses.

  | Config | What it compiles | Prefill / decode (cycles) |
  |---|---|---|
  | llama_cache | 1024-token prompt, residual 128 | 75,972,302 / 11,980,600 |
  | llama_cache_tail | 960 tokens, the 64-token tail into the residual | 71,602,963 / 11,980,600 |
  | llama_cache_sdpa | as tail, with sdpa attention | 70,449,581 / 11,846,973 |
  | llama_cache_blocks | 1024 tokens, no residual | 75,972,302 / 11,944,340 |

- **bert/MXNF4:** `--report` fails at this commit (see fd6feef).
- **Numerics** (the outputs printed by a failed check, compared between the
  two commits). The reasons below come from reading the code and the
  diffs; no build re-created the old setup to confirm them.
  - llama_prefill, _db: the same numbers. The compiled output is now a bare
    tensor instead of a one-element tuple.
  - llama_prefill_mp, _spmm, _spmm_db: changed, because the fp8 softmax and
    RMSNorm quantizes are gone. The uncompiled model's own output moves too
    (one logit −0.5156 → −0.4746), not only the compiled one.
  - The 4 decode configs: changed. The cache is now filled by the quantized
    prefill graph, where before the float HF model filled it; in MXNF4 the
    new key is also MX-quantized, and in _kivi the fp8 softmax and RMSNorm
    quantizes are gone. llama_decode/BF16 shows the cache is the cause: its
    `model.txt` is identical at both commits, yet its output changed.
  - llama_train: unchanged (BF16 and MXFP4 print the same numbers; FP8
    passes).
  - The new configs fail on the prefill logits, and the decode logits are
    not checked: 22.7% of elements mismatched in llama_cache and _blocks,
    31.2% in _tail, 81.5% in _sdpa.

## f9d4c49 → 0a94c31 (2026-09-14 to 2026-10-24, 65 commits)

The first entry. It was written by hand from comparing the CI output at
f9d4c49 with the output at 0a94c31. Each change was then pinned to a commit
by building `model.txt` at that commit and at its parent, unless the entry
says otherwise. Predicted latency was not recorded for these commits.

- **Not listed per commit:** 61% of the changed values are cosmetic. They
  come from renumbered node names, moved addresses, and kwargs and op names
  spelled differently. Addresses move whenever a real change below makes a
  buffer bigger or smaller.
- **Training configs:** f9d4c49 had no training configs, so the three
  `llama_train` configs are compared from the 2026-10-23 run on. That run
  was built from 71fccde's code (a cef1e7e checkout carrying 71fccde's
  edits); a 71fccde build reproduces it exactly.

### 81459bc fix: Voyager accumulates in fp32 where asked; ops carry their layout (2026-10-24)

- **model.txt:** layout text only, in all 26 configs shared with the
  2026-10-23 run: einsum strings such as `cx,ck` and the conv `layout:`
  field. Nothing else changed in those configs.
- **model.txt, the 3 `llama_train` gradient programs** (the only configs
  run with `--accumulate_fp32`): a GEMM step whose sum is split along K
  now returns its partial sum in fp32 (`out_dtype: "float32"`), so the
  separate `aten::to(float32)` before the add into the fp32 accumulator is
  gone. A bare `aten::matmul` becomes `quantized_ops::matmul`, which takes
  `out_dtype`. Ops changed: BF16 42, FP8 12, MXFP4 12. The final
  `to(bfloat16)` of the finished sum stays; shapes, sizes and addresses
  are unchanged, and the update programs did not change.
- **Numerics:** `--accumulate_fp32` keeps split-K partial sums in fp32
  (see f0592b4).

### 19367f4 feat: real RMSNorm, and Trainium kernels for the six-kernel suite (2026-10-14)

- **model.txt, all 8 Llama configs:** the norm is a real `aten::rms_norm`.
  Before, `replace_rmsnorm_with_layer_norm` swapped it for
  `aten.layer_norm`, which subtracts the mean first and so is a different
  computation. Llama prefill's 3 norm kernels keep their 32 tokens × 4096
  tile; the per-row scratch is now `mean_square` instead of `mean`.
- **Runtime model:** the norm is priced as 2 passes over the data (mean
  square, then normalize and scale) instead of LayerNorm's 4.
- **model.txt, llama_prefill_spmm and _spmm_db:** the norm tile grew from 16
  to 32 tokens × 4096, so the norm runs 64 steps instead of 128. Each norm
  tile writes one slice of the sparse outlier list (CSR), so each 256-token
  projection tile now loads 8 slices instead of 16. The projections' own
  tiles did not change.
- **Calibration:** calibration now runs through the real RMSNorm. The
  outlier threshold `quantize_mx_outlier_default_1` (the attention output,
  input to `o_proj`) moved one bf16 step, from 0.043212890625 to 0.04296875.
  The other three thresholds are unchanged.

### bd52d55 feat: Trainium backend, with the tile search on a per-backend runtime model (2026-10-13)

- **Bug fix, llama_decode, _db and _kivi:** RoPE's `emb = cat(freqs, freqs)`
  now writes both halves of its 128-value buffer. Before, a dedup pass merged
  the two identical stores and emit kept only the last, so offsets 0–63 were
  never written. Now a value has one destination, emit raises on a second
  store, and each half has its own `clone` (`clone_default` at offset 0,
  `clone_default_1` at 64). No other config's clone count changed.
- Placed by the CI runs of 2026-10-12 (1 clone) and 2026-10-13 (2 clones)
  and by this commit's emit diff; not rebuilt at its parent.

### 61c5ca1 feat: compile BERT and causal-LM training steps (2026-09-30)

- **model.txt, the split-K GEMVs (see f0592b4):** the separate bf16
  accumulator is gone. The partial sum goes straight into the output tile,
  with 2 op variants instead of 3. This covers Llama decode `k_proj`,
  `v_proj` and `down_proj`.

### 73f0ab5 feat: replace eager QAT with PT2E QAT (2026-09-27)

- **model.txt, the 10 per-tensor configs** (resnet and mobilebert with E4M3,
  INT8 or P8_1): residual adds are no longer quantized, so a block's output
  stays bf16.
  - The next block's conv still reads fp8 or int8, so a standalone quantize
    kernel makes that copy. resnet18 gains 4: `quantize_default_3`, `_7`,
    `_11` and `_15`.
  - The add reads the residual in bf16. In resnet18 E4M3 that is 401 KB per
    block instead of 200 KB.
  - In blocks with a downsample, the add moved from the downsample conv's
    kernel into `conv2`'s.
- The reason, from the commit: match eager QAT, which quantized only convs
  and linears. A quantized residual add cost 0.29 F1 on SQuAD E4M3 PTQ.

### f0592b4 fix: rank every valid L2 tiling when a cost model is available (2026-09-24)

- **model.txt, 16 configs:** GEMMs with a single input row (GEMVs) now split
  along K, the dimension they sum over. Each step reads part of the input row
  and adds its partial sum to the output. This covers the final
  `fc`/classifier (plus BERT's pooler) in resnet, vit, bert and mobilebert,
  and Llama decode `k_proj`, `v_proj` and `down_proj`. Example: decode
  `k_proj` went from 128 outputs × all 4096 inputs over 8 steps to all 1024
  outputs × 256 inputs over 16 steps. Not bit-exact against the unsplit
  version.
- **model.txt, vector ops in all configs:** the tile search tries every
  valid tile shape instead of shrinking from the whole tensor along one
  fixed path. 5 of resnet18/MXINT8's 7 vector kernels and 3 of Llama
  prefill's 11 changed, softmax among them. Example: Llama prefill RoPE
  `neg` went from 4 heads × 1024 tokens × 64 to 32 heads × 128 tokens × 64,
  8 steps both.
- Pools still take the largest tile that fits, unpriced.

### 6212bf6 fix: pick the least-DRAM tiling within tolerance and tidy sweep output (2026-09-23)

- **model.txt, GEMM and conv tiles:** among tilings within the tolerance of
  the fastest, the search now takes the one with the least DRAM traffic.
  Retiled: 6 of resnet18/MXINT8's 21 GEMM/conv kernels, 1 of Llama
  prefill's 10.

### 376a353 fix: price the tail's scale-bank burst and the direct-path drain (2026-09-19)

- **model.txt, GEMM and conv tiles:** the runtime model charges a stall the
  RTL showed, where the output burst starves the scale reads. Retiled:
  resnet18/MXINT8 3, Llama prefill 2. Example: Llama prefill `o_proj` went
  from 256 tokens × 128 columns over 128 steps to 64 tokens × 256 columns
  over 256 steps.

### 34ceff2 fix: price a matrix tile as the RTL runs it and prefer fewer ops among equals (2026-09-17)

- **model.txt, GEMM and conv tiles:** matrix tiles are priced to match the
  RTL, and among tilings within 2% of the fastest the search takes the one
  with the fewest steps. Retiled: resnet18/MXINT8 8, Llama prefill 1.
  Example: resnet18 `layer1_1_conv2` went from a 58×10 strip with all 64
  output channels over 7 steps to the whole 58×58 map with 16 output
  channels over 4 steps.
- **model.txt, resnet E4M3/32x64:** the layer4 3×3 convs no longer split
  along K. They read all 512 input channels × 64 output channels over 8
  steps, instead of 128 × 128 over 16.
- **model.txt, mobilebert (E4M3, INT8, P8_1) and vit:** Q and K projection
  tiles now hold one attention head, so reordering the tile into head-first
  order moves no data and becomes a view. The transpose op disappears from
  Q and K; V's tile now holds 2 heads and does its transpose in the kernel.
  In vit the separate transpose-and-quantize kernel is gone; the quantize
  fuses into the projection.

### 2c1c1be fix: price vector passes by what each pass reads and writes (2026-09-15)

- **model.txt, vector-op tiles:** each pass of a vector op is charged for
  what it actually reads and writes, matched to the RTL. Before, every pass
  paid a full read at the widest operand width. Example: resnet18
  `quantize_mx_default_10` went from 7×14×256 over 2 steps to 2×14×256 over
  7 steps. resnet18's final `fc` also retiled.

Tile changes were traced commit by commit only in resnet18/MXINT8 and Llama
prefill (plus mobilebert/INT8, vit and resnet18/E4M3/32x64 for 34ceff2).
The other configs come from the same tile search but were not built at
every commit.
