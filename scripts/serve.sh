#!/bin/bash
# Serve the base model and, optionally, a trained adapter with vLLM for evaluation.
# Run it inside the training environment (the one with vLLM).
#
#   MODEL=/path/to/Qwen3.6-27B [ADAPTER=/path/to/checkpoints/global_step_12/actor/lora_adapter] \
#   [GPUS=0,1,2,3] [PORT=18000] bash scripts/serve.sh
#
# Served names: "policy" is the base model; "trained" is the base model with the adapter.
# Sampling is at temperature 1.0 without truncation, as in training.
set -euo pipefail
: "${MODEL:?set MODEL}"
GPUS=${GPUS:-0,1,2,3}; PORT=${PORT:-18000}
IFS=',' read -r -a G <<< "$GPUS"
LORA=()
if [ -n "${ADAPTER:-}" ]; then
  [ -f "$ADAPTER/adapter_config.json" ] || { echo "no adapter at $ADAPTER" >&2; exit 1; }
  LORA=(--enable-lora --max-lora-rank 32 --lora-modules "trained=$ADAPTER")
fi
EXTRA=()
[ -n "${GDN_PREFILL_BACKEND:-}" ] && EXTRA=(--additional-config "{\"gdn_prefill_backend\":\"$GDN_PREFILL_BACKEND\"}")
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0} VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
CUDA_VISIBLE_DEVICES=$GPUS exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name policy --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --load-format safetensors \
  --max-model-len 262144 --max-num-seqs "${MAX_NUM_SEQS:-64}" --tensor-parallel-size "${#G[@]}" \
  --gpu-memory-utilization 0.85 --seed "${SEED:-0}" ${LORA[@]+"${LORA[@]}"} \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 ${EXTRA[@]+"${EXTRA[@]}"} \
  --override-generation-config '{"temperature":1.0,"top_k":-1,"top_p":1.0,"repetition_penalty":1.0}'
