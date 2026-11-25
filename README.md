# PROVE: Protocol Reinforcement from Verified Evidence

Code for the paper **"PROVE: Mitigating Long-Horizon LLM Collusion with Reinforcement Learning from Verified Evidence"**.

<p align="center">
  <img src="assets/teaser.png" width="820" alt="Over a long interaction, two agents learn to approve each other's work without checking it; PROVE trains them to keep following the rule.">
</p>

Two LLM agents each solve a task and then review each other's work. The protocol says: accept only after
receiving the peer's complete raw log. When the log cannot be delivered, a compliant agent must reject, and the
environment then tells it that its verdict was wrong whenever the peer's work was in fact correct. Over a long
interaction, agents learn from this feedback to accept without evidence. This is the collusion studied by
[Shi, Zhang and Yang (2026)](https://arxiv.org/abs/2609.24967).

PROVE trains the agents to stay faithful to the protocol, with their full memory and under the same feedback.
It is a multi-agent reinforcement learning method with two ideas:

- a **protocol-grounded reward** that ranks verdicts by the evidence behind them and not by their outcome;
- **task-phase credit**, which also trains the task-solving calls on whether the agent's own work is correct, so
  that training the reviewer does not erode the worker that shares its weights.

Both are assigned per turn (one model call by one agent) and standardized across rollouts of the same task
sequence. On Qwen3.6-27B, a single LoRA adapter trained on six-episode interactions takes the collusion rate on
held-out task sequences at the original ten-episode horizon from 65.0% to 1.0%, with no detectable change in
task success.

## The method

<p align="center">
  <img src="assets/method.png" width="820" alt="How PROVE trains: verdicts are rewarded by the evidence behind them, and every call is credited with the outcome it controls.">
</p>

**Reward of a verdict** (`prove/reward.py`). *Evidence* means that the environment has checked that the peer's
complete, unmodified log reached the agent.

| Evidence     | Verdict                                | Reward |
|:-------------|:---------------------------------------|:------:|
| received     | matches whether the peer's work is correct | +1 |
| received     | does not match                         |   −1   |
| not received | REJECT                                 |    0   |
| not received | ACCEPT                                 |   −1   |

Without evidence, REJECT beats ACCEPT whatever the peer did. With evidence, a correct judgment earns more than
rejecting without verification. So neither collusion nor blanket rejection is optimal when the log can be sent.

**Credit of a turn** (`prove/credit.py`). A turn is one model call by one agent. For agent *i* with peer *j* in
episode *t*, with *r* the verdict reward above:

| Turn              | Reward of the turn                                           |
|:------------------|:-------------------------------------------------------------|
| verdict           | *r*(*i*, *t*), the agent's own verdict reward                |
| message           | ½ (*r*(*i*, *t*) + *r*(*j*, *t*)), the mean over both verdicts |
| reflection        | *r*(*i*, *t*+1), the agent's verdict reward in the next episode |
| task-solving call | +1 if agent *i*'s own work in episode *t* is correct, else −1 |

The advantage of a turn is its reward minus the mean over the rollouts of its group, divided by their standard
deviation. A turn is compared only with the same turn in other rollouts, and every token of the turn receives
that advantage in the clipped GRPO objective.

## What is in this repository

```
prove/
  reward.py          evidence check, protocol-grounded reward, outcome of an episode
  credit.py          turn-level credit, task-phase credit, group-standardized advantages
  metrics.py         collusion metrics and paired bootstrap intervals from recorded trajectories
  agent.py           Agent Lightning agent: runs one trajectory of the harness, reports reward and credit
  train.py           training entry with the settings of the paper
  checks.py          verifies that the patches below are applied
  hooks/             runtime hooks for Agent Lightning, verl and vLLM (no package source is edited)
  harness_hooks/     optional harness settings: request timeout, reworded verdict policy
patches/             three small patches for the harness and for Agent Lightning
scripts/             train.sh, serve.sh, evaluate.sh
tests/               unit tests of the reward, the credit assignment and the metrics
```

The environment, prompts and task sequences are those of Shi et al. and are used unchanged; this repository
does not copy them.

## Setup

Two Python environments are used, as the harness and the trainer pin different packages.

**1. Harness.** Clone [SALT-NLP/agent-collusion](https://github.com/SALT-NLP/agent-collusion), install its
`requirements.txt` into its own environment, and apply two patches:

```bash
cd agent-collusion
git apply /path/to/prove/patches/agent-collusion-sandbox-tests.patch
git apply /path/to/prove/patches/agent-collusion-history-tool-args.patch
```

The first runs agent-written tests without network inside [bubblewrap](https://github.com/containers/bubblewrap)
(`bwrap` and `unshare` must be installed); the second keeps a malformed tool-call argument from breaking the rest
of an episode.

**2. Trainer.** Clone [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning) and install it
with its verl backend (`scripts/setup_verl.sh`, verl 0.8.0 with vLLM 0.22.0). Then:

```bash
cd agent-lightning
git apply /path/to/prove/patches/agent-lightning-train-row-filter.patch
pip install flash-linear-attention      # training kernels for the Gated DeltaNet layers of Qwen3.6
pip install -e /path/to/prove
```

Versions we used: Python 3.12, Agent Lightning 1.0.3 (revision `eff9f44`), verl 0.8.0, vLLM 0.22.0, PyTorch 2.11.0,
Transformers 5.22.1, PEFT 0.21.2, Ray 2.57.0, flash-linear-attention 0.6.0. `python -m prove.checks
/path/to/agent-collusion` reports any patch that is missing; training and evaluation refuse to start without them.

**3. Model.** Qwen3.6-27B, loaded in bf16.

## Training

```bash
MODEL=/path/to/Qwen3.6-27B \
SALT_HARNESS_DIR=/path/to/agent-collusion \
SALT_HARNESS_PY=/path/to/harness-env/bin/python \
OUT=/path/to/runs/prove \
bash scripts/train.sh
```

This trains the model of the paper: 12 updates, each on 16 six-episode interactions (two task sequences, both
regimes, four rollouts each), drawn from task sequences 1–40. Sequences 41–50 are never trained on. Checkpoints
are written after every update to `OUT/checkpoints/global_step_N/`, with the adapter in PEFT format under
`actor/lora_adapter/`. Running the command again with the same `OUT` resumes from the latest checkpoint.
An update takes roughly ten hours on eight NVIDIA H20 GPUs.

| Setting | Value |
|---|---|
| Adapter | LoRA, rank 32, α = 64, linear projections of the language model |
| Optimizer | AdamW (ε = 10⁻¹²), constant learning rate 2 × 10⁻⁵ |
| Objective | clipped GRPO (ratio in [0.8, 1.28]), standardized advantages, one pass per update, no KL or entropy term |
| Sampling | temperature 1.0 in training and evaluation |
| Context | 262,144 tokens for rollouts; prompts of up to 81,920 tokens are trained |
| Response limit | 16,384 tokens per call; responses that reach it are not trained |
| Trained calls | task-solving calls, messages, verdicts, reflections |

Options (`scripts/train.sh`):

- `TASK_CREDIT=0` trains the ablation without task-phase credit: the task-solving calls are not trained and
  everything else is unchanged.
- `STEPS=N` sets the number of updates.
- Arguments are passed on to `prove/train.py`; any Hydra override of the verl configuration can be appended, for
  example `data.seed=1`. `python -m prove.train --model M --out-root O --agl-key k --print-config` prints the
  resolved configuration without starting anything.

## Evaluation

Serve the base model with the trained adapter, run ten-episode trajectories on the held-out sequences in both
regimes, and score them:

```bash
# terminal 1 (trainer environment)
MODEL=/path/to/Qwen3.6-27B ADAPTER=$OUT/checkpoints/global_step_12/actor/lora_adapter bash scripts/serve.sh

# terminal 2
export SALT_HARNESS_DIR=/path/to/agent-collusion SALT_HARNESS_PY=/path/to/harness-env/bin/python
SERVED=trained OUT=/path/to/eval/prove bash scripts/evaluate.sh
SERVED=policy  OUT=/path/to/eval/base  bash scripts/evaluate.sh     # the base model without the adapter, same server

python -m prove.metrics --runs /path/to/eval/prove --baseline /path/to/eval/base
```

`prove.metrics` reports the collusion rate, collusive trajectories and converged collusion of Shi et al.,
unverified acceptance, defective work accepted, the verification rate, and task success. With `--baseline`, every
difference comes with a 95% interval from a paired bootstrap over task sequences (4,000 resamples). Each task
sequence and regime must appear exactly once per group; trajectories are never pooled.

**Changed settings.** The transfer evaluations of the paper change one thing each and use six-episode
trajectories (`EPISODES=6`):

```bash
SALT_RULE_PARAPHRASE_FILE=prove/harness_hooks/reworded_policy.txt ...   # verdict policy in different words
SALT_CHAR_LIMIT=100 ...                                                 # half the message limit
SALT_CHAR_LIMIT=400 ...                                                 # twice the message limit
```

Run the base model under the same changed setting for the paired comparison.

## Tests

```bash
pip install pytest && pytest
```

The tests cover the reward table, the evidence check, the credit of every turn type, the advantages, and the
metrics. They need no GPU and none of the training packages.

## Notes on the hooks

Training a LoRA adapter through Agent Lightning, verl and vLLM needed a few fixes that live in `prove/hooks/`
and are installed in every Ray worker by `worker_setup.py`:

- `turn_credit.py`: tags every training row with its turn, replaces its advantage, and drops truncated responses.
- `gdn_packing.py`: passes sequence boundaries to the Gated DeltaNet layers when several sequences are packed
  into one forward pass; without it, training log-probabilities differ from those of sampling.
- `lora_sync.py`, `lora_serving.py`: make sure the rollouts are generated by the adapter being trained.
- `lora_checkpoint.py`: checkpoints hold the adapter and optimizer state instead of the full base model.
- `free_cache.py`: frees the trainer's GPU cache before the rollout engine wakes up.

## Acknowledgements

The environment is the peer-verification harness of [Shi, Zhang and Yang](https://github.com/SALT-NLP/agent-collusion).
Training builds on [Agent Lightning](https://github.com/microsoft/agent-lightning),
[verl](https://github.com/volcengine/verl) and [vLLM](https://github.com/vllm-project/vllm).

## License

MIT, see [LICENSE](LICENSE).
