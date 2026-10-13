#!/bin/bash
# The AWS baseline: NxD Inference at the Trainium design points.
#
#   nxdi_baseline.sh compile <model dir> <model type> <out dir>
#   nxdi_baseline.sh benchmark <model dir> <model type> <out dir>
#
# compile builds the context-encoding (prefill) and token-generation
# (decode) NEFFs for the target NEURON_PLATFORM_TARGET_OVERRIDE names (trn2
# or trn3) and needs only the model's config: it runs on a CPU-only
# machine.  benchmark loads them on a NeuronCore of that target with the
# checkpoint's weights and writes benchmark_report.json (prefill and decode
# latency percentiles).  Source ~/claude/trn/env.sh first (Neuron SDK
# 2.31.1), then set the target: env.sh sets trn3.
#
# Design point: batch 1 on one NeuronCore (no tensor parallelism), a
# 1024-token context, 128 generated tokens, BF16, greedy on-device sampling.
set -euo pipefail

mode=$1
model_dir=$2
model_type=$3
out_dir=$4

CONTEXT=1024
NEW_TOKENS=128

common=(
    --model-type "$model_type" --task-type causal-lm run
    --model-path "$model_dir"
    --compiled-model-path "$out_dir"
    --torch-dtype bfloat16
    --tp-degree 1 --logical-nc-config 1
    --batch-size 1
    --max-context-length "$CONTEXT"
    --seq-len $((CONTEXT + NEW_TOKENS))
    --on-device-sampling
    --prompt "The"
)

case "$mode" in
    compile)
        inference_demo "${common[@]}" --compile-only
        ;;
    benchmark)
        inference_demo "${common[@]}" --skip-compile --benchmark \
            --benchmark-report-path "$out_dir/benchmark_report.json"
        ;;
    *)
        echo "usage: $0 {compile|benchmark} <model dir> <type> <out dir>"
        exit 2
        ;;
esac
