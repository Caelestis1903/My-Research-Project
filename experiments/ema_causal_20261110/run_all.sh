#!/usr/bin/env bash
set -euo pipefail
EMA_CAUSAL_PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
EMA_CAUSAL_PLAN=research_plans/26-11-10_ema_causal_attribution
EMA_CAUSAL_OUT=${1:?pass a new attempt directory}
for EMA_CAUSAL_COMMAND in audit prepare; do
  "$EMA_CAUSAL_PY" -m experiments.ema_causal_20261110.run "$EMA_CAUSAL_COMMAND" --plan-dir "$EMA_CAUSAL_PLAN" --out "$EMA_CAUSAL_OUT"
done
for EMA_CAUSAL_DATA in movielens kuairand; do
  "$EMA_CAUSAL_PY" -m experiments.ema_causal_20261110.run smoke --plan-dir "$EMA_CAUSAL_PLAN" --out "$EMA_CAUSAL_OUT" --dataset "$EMA_CAUSAL_DATA"
done
"$EMA_CAUSAL_PY" -m experiments.ema_causal_20261110.run run --plan-dir "$EMA_CAUSAL_PLAN" --out "$EMA_CAUSAL_OUT" --devices 0 1 --max-workers 2
for EMA_CAUSAL_STAGE in 1 2 3 4; do
  "$EMA_CAUSAL_PY" -m experiments.ema_causal_20261110.run analyze --plan-dir "$EMA_CAUSAL_PLAN" --out "$EMA_CAUSAL_OUT" --stage "$EMA_CAUSAL_STAGE"
done
# The orchestrator writes scientific findings while the independent guardian retains endpoints.
while [ ! -f "$EMA_CAUSAL_OUT/MECHANISM_FINDINGS.json" ]; do sleep 10; done
for EMA_CAUSAL_COMMAND in freeze evaluate report release; do
  "$EMA_CAUSAL_PY" -m experiments.ema_causal_20261110.run "$EMA_CAUSAL_COMMAND" --plan-dir "$EMA_CAUSAL_PLAN" --out "$EMA_CAUSAL_OUT"
done
