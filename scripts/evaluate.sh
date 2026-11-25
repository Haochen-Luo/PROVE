#!/bin/bash
# Run evaluation trajectories of the harness against a served model (scripts/serve.sh), then check that every
# trajectory is complete. Score the output with `python -m prove.metrics`.
#
#   SALT_HARNESS_DIR=/path/to/agent-collusion SALT_HARNESS_PY=/path/to/harness-env/bin/python \
#   OUT=/path/to/eval/prove [SERVED=trained] [PORT=18000] bash scripts/evaluate.sh
#
# SERVED     model name on the server: "trained" (base model + adapter) or "policy" (base model without the adapter)
# EPISODES   episodes per trajectory (default 10; the harness ships 50xT sequence records for T = 3, 6, 10)
# SEQS       task sequences (default: the held-out sequences 41-50)
# REGIMES    "throttled no-throttle" = constrained and unconstrained (default both)
# JOBS       trajectories run in parallel
# Changed settings for transfer evaluation:
#   SALT_CHAR_LIMIT=100 or 400            message limit of the constrained regime (default 200)
#   SALT_RULE_PARAPHRASE_FILE=<file>      reworded verdict policy (prove/harness_hooks/reworded_policy.txt)
set -euo pipefail
: "${SALT_HARNESS_DIR:?set SALT_HARNESS_DIR}" "${SALT_HARNESS_PY:?set SALT_HARNESS_PY}" "${OUT:?set OUT}"
SERVED=${SERVED:-trained}; PORT=${PORT:-18000}; EPISODES=${EPISODES:-10}
SEQS=${SEQS:-$(seq 41 50)}; REGIMES=${REGIMES:-throttled no-throttle}; JOBS=${JOBS:-20}
REPO=$(cd "$(dirname "$0")/.." && pwd)
[ -e "$OUT" ] && { echo "$OUT exists; choose a new OUT" >&2; exit 1; }

python -m prove.checks "$SALT_HARNESS_DIR" --harness-only
curl -s --max-time 10 "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$SERVED\"" \
  || { echo "no model named $SERVED on port $PORT" >&2; exit 1; }
mkdir -p "$OUT"

# Per-request limit of the harness's model calls, in seconds; long answers need it.
export SALT_LLM_TIMEOUT=${SALT_LLM_TIMEOUT:-1800}
export PYTHONPATH="$REPO/prove/harness_hooks${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${SALT_RULE_PARAPHRASE_FILE:-}" ]; then
  SALT_RULE_PARAPHRASE=$(cat "$SALT_RULE_PARAPHRASE_FILE"); export SALT_RULE_PARAPHRASE
  cp "$SALT_RULE_PARAPHRASE_FILE" "$OUT/reworded_policy.txt"
fi

run_one() {  # SEQ REGIME
  local s=$1 regime=$2 d extra=()
  [ -n "${SALT_CHAR_LIMIT:-}" ] && extra+=(--char-limit "$SALT_CHAR_LIMIT")
  d=$OUT/seq${s}_$regime; mkdir -p "$d"; cd "$SALT_HARNESS_DIR"
  env COLLUSION_SANDBOX_TESTS=1 OPENAI_API_BASE="http://127.0.0.1:$PORT/v1" OPENAI_API_KEY=local "$SALT_HARNESS_PY" -m experiments \
    --alice-model "openai/$SERVED" --bob-model "openai/$SERVED" --alice-reasoning-effort default --bob-reasoning-effort default \
    --alice-max-output-tokens 16384 --bob-max-output-tokens 16384 --task-sequence-record "task/task_sequences_50x$EPISODES" \
    --start-index "$s" --repeats 1 --throttle-spans "$regime:*" ${extra[@]+"${extra[@]}"} --output-dir "$d" --quiet --no-preflight > "$d/harness.log" 2>&1 \
    || echo "FAILED $s $regime"
}
export -f run_one; export OUT SERVED PORT EPISODES SALT_HARNESS_DIR SALT_HARNESS_PY
for s in $SEQS; do for regime in $REGIMES; do printf '%s %s\n' "$s" "$regime"; done; done \
  | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _ | tee "$OUT/failures.txt"

# Every (sequence, regime) must have exactly one trajectory with all episodes.
SEQS="$SEQS" REGIMES="$REGIMES" python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["OUT"])
incomplete = []
for seq in os.environ["SEQS"].split():
    for regime in os.environ["REGIMES"].split():
        files = list((root / f"seq{seq}_{regime}").glob("run_*/run.json"))
        if len(files) != 1 or len(json.loads(files[0].read_text())["results"]) != int(os.environ["EPISODES"]):
            incomplete.append(f"seq{seq}_{regime}")
print({"incomplete": incomplete})
if incomplete or (root / "failures.txt").stat().st_size:
    raise SystemExit("evaluation has failed or incomplete trajectories")
PY
echo "done: $(find "$OUT" -name run.json | wc -l) trajectories in $OUT"
