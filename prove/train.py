"""Training entry: PROVE on the peer-verification environment with Agent Lightning (verl backend).

One shared LoRA policy plays both agents. A rollout is one multi-episode trajectory of a task sequence in one
regime; every training batch holds both regimes of the same sequences, and each (sequence, regime) is sampled
``rollout.n`` times to form a GRPO group. The reward is ``prove.reward``; advantages are assigned per turn by
``prove.hooks.turn_credit``.

Start it through ``scripts/train.sh``, which launches Ray, the Agent Lightning server and controller, and sets
the environment variables the hooks read. The defaults below are the settings of the paper (Qwen3.6-27B, eight
GPUs, six-episode sequences); any Hydra override can be appended on the command line.
"""
from __future__ import annotations

import argparse
import importlib.resources
import os
import random
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REGIMES = ["throttled:*", "no-throttle:*"]  # constrained, unconstrained (the harness's names)


def rows(seqs, regimes, manifests):
    return [{"seq": s, "throttle_spans": r, "manifests": manifests} for s in seqs for r in regimes]


def overrides(model, n_gpus, rollout_tp, epochs, out_root):
    env_map = {"SALT_SEQ": "input.seq", "SALT_THROTTLE_SPANS": "input.throttle_spans", "SALT_MANIFESTS": "input.manifests"}
    token_budget = 98304  # per GPU and micro-batch; must be >= max_prompt_length + max_response_length
    return {
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False, "enable_rollout_level_advantage": True},
        "data": {"train_batch_size": 4, "max_prompt_length": 81920, "max_response_length": 16384},
        "actor_rollout_ref": {
            "model": {"path": model, "use_remove_padding": True, "enable_gradient_checkpointing": True,
                      "lora_rank": 32, "lora_alpha": 64,
                      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      "exclude_modules": ".*visual.*",
                      "use_fused_kernels": True, "fused_kernel_options": {"impl_backend": "triton"}},
            "rollout": {"name": "vllm", "n": 4, "tensor_model_parallel_size": rollout_tp, "gpu_memory_utilization": 0.55,
                        "log_prob_micro_batch_size_per_gpu": 1, "multi_turn": {"format": "hermes"},
                        "log_prob_use_dynamic_bsz": True, "log_prob_max_token_len_per_gpu": token_budget,
                        "max_model_len": 262144,
                        # LoRA mode: real weights in vLLM, adapter synced layer by layer after every update.
                        "load_format": "safetensors", "layered_summon": True},
            "actor": {"ppo_mini_batch_size": 4, "ppo_micro_batch_size_per_gpu": 1,
                      # Agent Lightning's per_rollout_mean loss divides by rollout tokens and trained rows, which
                      # puts gradients below AdamW's default eps of 1e-8; 1e-12 keeps Adam scale invariant.
                      "optim": {"lr": 2e-5, "override_optimizer_config": {"eps": 1e-12}},
                      "use_dynamic_bsz": True, "ppo_max_token_len_per_gpu": token_budget,
                      "entropy_from_logits_with_chunking": True, "entropy_checkpointing": True,
                      "use_kl_loss": False, "kl_loss_coef": 0.0, "entropy_coeff": 0,
                      "clip_ratio_low": 0.2, "clip_ratio_high": 0.28,
                      "policy_loss": {"loss_mode": "per_rollout_mean"},
                      "fsdp_config": {"param_offload": False, "optimizer_offload": False}},
            "ref": {"log_prob_micro_batch_size_per_gpu": 1, "fsdp_config": {"param_offload": True},
                    "log_prob_use_dynamic_bsz": True, "log_prob_max_token_len_per_gpu": token_budget},
        },
        "trainer": {"n_gpus_per_node": n_gpus, "nnodes": 1, "val_before_train": False, "critic_warmup": 0,
                    "logger": ["console"], "project_name": "prove", "experiment_name": "prove",
                    "save_freq": 1, "test_freq": 2, "max_actor_ckpt_to_keep": 5, "total_epochs": epochs,
                    "default_local_dir": f"{out_root}/checkpoints"},
        # Installs the hooks of prove.hooks in every Ray worker.
        "ray_kwargs": {"ray_init": {"runtime_env": {
            "worker_process_setup_hook": "prove.hooks.worker_setup.register_in_worker"}}},
        "agentlightning": {
            "rollout_timeout_seconds": 18000,
            "max_ppo_update_times": None,  # train every collected row
            # One row per model call: the two agents' calls interleave, so each call is its own row.
            "trace_aggregator": {"level": "transition"},
            "local": {"agent_class": "prove.agent.PeerVerificationAgent", "env_map": env_map},
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="path of the base model")
    p.add_argument("--out-root", required=True, help="run directory (checkpoints, credit files, resolved config)")
    p.add_argument("--agl-base-url", default="http://localhost:8181")
    p.add_argument("--agl-key", required=True)
    p.add_argument("--n-gpus", type=int, default=8)
    p.add_argument("--rollout-tp", type=int, default=4, help="tensor-parallel size of each vLLM rollout replica")
    p.add_argument("--epochs", type=int, default=10, help="upper bound; end a run with trainer.total_training_steps=N")
    p.add_argument("--manifests", default="task/task_sequences_50x6",
                   help="task-sequence record of the harness; 50xT holds 50 sequences of T episodes")
    p.add_argument("--train-seqs", default="1:41:1", help="start:stop:step over task sequences used for training")
    p.add_argument("--val-seqs", default="41:51:1", help="start:stop:step over held-out task sequences")
    p.add_argument("--val-regimes", default=",".join(REGIMES), help="regimes of the validation pass")
    p.add_argument("--tool-call-parser", default="qwen3_coder", help="vLLM tool-call parser of the model")
    p.add_argument("--gdn-prefill-backend", default=None, choices=["triton", "flashinfer", "cutedsl"],
                   help="vLLM Gated DeltaNet prefill backend, if the default does not run on your driver")
    p.add_argument("--print-config", action="store_true", help="print the resolved configuration and exit")
    p.add_argument("overrides", nargs="*", help="extra Hydra dotlist overrides")
    args = p.parse_intermixed_args()

    with initialize_config_dir(config_dir=str(importlib.resources.files("agentlightning.verl")), version_base=None):
        cfg = compose(config_name="config")
    OmegaConf.set_struct(cfg, False)  # the overrides add keys that the base schema does not list
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides(args.model, args.n_gpus, args.rollout_tp, args.epochs, args.out_root)))
    cfg.agentlightning.agl_base_url, cfg.agentlightning.agl_key = args.agl_base_url, args.agl_key
    cfg.actor_rollout_ref.rollout.engine_kwargs.vllm.update(
        {"enable_auto_tool_choice": True, "tool_call_parser": args.tool_call_parser, "reasoning_parser": "qwen3"})
    if args.gdn_prefill_backend:
        cfg.actor_rollout_ref.rollout.engine_kwargs.vllm.update(
            {"additional_config": {"gdn_prefill_backend": args.gdn_prefill_backend}})
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    OmegaConf.set_struct(cfg, True)

    # Paired regimes: a fixed-seed shuffle of the sequences, rows kept as [constrained, unconstrained] pairs and
    # read in order, so every batch contains both regimes of the same sequences.
    batch_size = int(cfg.data.train_batch_size)
    if batch_size % 2:
        raise SystemExit(f"data.train_batch_size must be even, got {batch_size}")
    train_seqs = list(range(*map(int, args.train_seqs.split(":"))))
    random.Random(int(cfg.data.get("seed") or 0)).shuffle(train_seqs)
    cfg.data.shuffle = False
    train = rows(train_seqs, REGIMES, args.manifests)
    val_regimes = [x.strip() for x in args.val_regimes.split(",") if x.strip()]
    if not set(val_regimes) <= set(REGIMES):
        raise SystemExit(f"--val-regimes must be a subset of {REGIMES}, got {val_regimes}")
    val = rows(range(*map(int, args.val_seqs.split(":"))), val_regimes, args.manifests)
    print(f"[prove] train rows {len(train)} | validation rows {len(val)}")

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    (Path(args.out_root) / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
        return
    from prove.checks import check

    check(os.environ["SALT_HARNESS_DIR"])
    from agentlightning.verl.entrypoint import run_ppo

    run_ppo(cfg, train_dataset=train, val_dataset=val)


if __name__ == "__main__":
    main()
