#!/bin/bash
# Train PROVE on one node: Ray, the Agent Lightning server and controller, and the training entry.
# Run it inside the training environment (README, "Setup").
#
#   MODEL=/path/to/Qwen3.6-27B \
#   SALT_HARNESS_DIR=/path/to/agent-collusion SALT_HARNESS_PY=/path/to/harness-env/bin/python \
#   OUT=/path/to/runs/prove bash scripts/train.sh [train.py options] [Hydra overrides]
#
# STEPS=12          number of updates (the paper's model: 12)
# TASK_CREDIT=0     ablation: do not train the calls of the task phase
# Re-running with the same OUT resumes from the latest checkpoint in OUT/checkpoints.
set -euo pipefail
: "${MODEL:?set MODEL}" "${SALT_HARNESS_DIR:?set SALT_HARNESS_DIR}" "${SALT_HARNESS_PY:?set SALT_HARNESS_PY}" "${OUT:?set OUT}"
PORT=${AGL_SERVER_PORT:-8181}
KEY=${AGL_KEY:-prove-$(date +%s)}
STEPS=${STEPS:-12}
LOGDIR=$OUT/logs/$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR" "$OUT/rollouts"

export MODEL SALT_HARNESS_DIR SALT_HARNESS_PY
export SALT_OUT_ROOT=$OUT/rollouts PROVE_CREDIT_DIR=$OUT/credit
export PROVE_TASK_CREDIT=${TASK_CREDIT:-1}
# The settings of our runs; they select vLLM code paths that do not need kernels compiled at run time.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0} VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}

# Which model calls enter training. Agent Lightning searches the whole response, which starts with the
# thinking text, so the answer is matched after the LAST </think>. With task-phase credit every non-empty
# answer passes; without it, answers that open a tool call ('[' or '<') are left out. The hook in
# prove/hooks/turn_credit.py then keeps exactly the rows of the trained turn types, in the form each requires.
if [ "$PROVE_TASK_CREDIT" = 1 ]; then
  export AGL_TRAIN_RESPONSE_REGEX='(?s)^(?:.*</think>(?!.*</think>)|(?!.*</think>))\s*\S'
else
  export AGL_TRAIN_RESPONSE_REGEX='(?s)^(?:.*</think>(?!.*</think>)|(?!.*</think>))\s*[^\s\[<]'
fi

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do [ -n "$pid" ] && kill "$pid" 2>/dev/null || true; done
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM
echo "logs: $LOGDIR"

# Ray prestarts one idle worker per CPU; capping the count keeps the thread use of a large machine bounded.
ray start --head --dashboard-host=127.0.0.1 --num-cpus="${RAY_NUM_CPUS:-32}" \
  ${RAY_NODE_IP:+--node-ip-address=$RAY_NODE_IP} > "$LOGDIR/ray.log" 2>&1
# Request limit between the Agent Lightning proxy and the model server, in seconds.
agl-server port="$PORT" key="$KEY" default_proxy.model_name="$MODEL" \
  upstream_timeout_seconds="${AGL_UPSTREAM_TIMEOUT:-300}" > "$LOGDIR/agl-server.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do curl -sf "http://localhost:$PORT/healthz" >/dev/null && break; sleep 1; done
agl-controller runner_type=local agl_server.url="http://localhost:$PORT" agl_server.key="$KEY" \
  > "$LOGDIR/agl-controller.log" 2>&1 &
PIDS+=($!)

python -m prove.train --model "$MODEL" --out-root "$OUT" --agl-base-url "http://localhost:$PORT" --agl-key "$KEY" \
  trainer.total_training_steps="$STEPS" "$@" 2>&1 | tee "$LOGDIR/train.log"
