#!/usr/bin/env sh
# Real-model benchmark: travel-desk + BFCL v4 subsets + judge calibration on MT-Bench human labels.
#
#   BASE_URL=http://127.0.0.1:8081/v1 MODEL=qwen2.5-0.5b-instruct-q4_k_m TAG=qwen0.5b sh scripts/run_benchmark.sh
#
# Start the model first, e.g. llama.cpp:
#   llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --jinja -c 8192 -t 2 --port 8081
# Inputs: data/bfcl (agentlens bfcl <category>) and data/mtbench (python scripts/prepare_mtbench.py).
set -eu
BASE_URL=${BASE_URL:-http://127.0.0.1:8081/v1}
MODEL=${MODEL:-qwen2.5-0.5b-instruct-q4_k_m}
TAG=${TAG:-qwen0.5b}
STEPS=${STEPS:-travel bfcl judge}
OUT=${OUT:-results}
mkdir -p "$OUT/traces"
A="--model $MODEL --base-url $BASE_URL --label $TAG --quiet"

for step in $STEPS; do
  case $step in
    travel)
      agentlens eval examples/travel_desk/suite.yaml $A \
        --trace "$OUT/traces/travel_desk_$TAG.jsonl" --out "$OUT/travel_desk_$TAG.json" --report "$OUT/travel_desk_$TAG.md" ;;
    bfcl)
      for spec in simple_python:100 multiple:50 irrelevance:50; do
        cat=${spec%%:*}; n=${spec##*:}
        agentlens eval "data/bfcl/bfcl_$cat.json" --limit "$n" $A \
          --trace "$OUT/traces/bfcl_${cat}_$TAG.jsonl" --out "$OUT/bfcl_${cat}_$TAG.json" --report "$OUT/bfcl_${cat}_$TAG.md"
      done ;;
    judge)
      agentlens judge data/mtbench/human_pairs.jsonl --mode pairwise --model "$MODEL" --base-url "$BASE_URL" \
        --max-tokens 384 --quiet --out "$OUT/judge_mtbench_$TAG.json" --predictions "$OUT/judge_mtbench_${TAG}_predictions.jsonl" ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
  echo "done: $step ($(date -u +%H:%M:%S))"
done
