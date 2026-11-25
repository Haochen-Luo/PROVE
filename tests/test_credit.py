import math
import os

from prove.credit import (DROP, load_credit, response_shape_ok, rollout_id_from_event_url, tag_prompt, trained_types,
                          turn_advantages, turn_credit, write_credit)

REWARDS = [{"alice": 0.0, "bob": -1.0}, {"alice": 1.0, "bob": 0.0}, {"alice": -1.0, "bob": -1.0}]
TASK_OK = [{"alice": True, "bob": False}, {"alice": True, "bob": True}, {"alice": False, "bob": True}]


def test_tag_prompt():
    system = "You are Alice. You work with Bob.\n"
    assert tag_prompt(system + "## Episode 1: task phase\n...\n## Episode 2: communication phase\nSend a message.") == (
        2, "alice", "communication")
    assert tag_prompt(system + "## Episode 3: verdict phase\nSubmit your verdict.") == (3, "alice", "verdict")
    text = system + "## Episode 3: verdict phase\n...\nRespond with a brief plain-text reflection on this episode."
    assert tag_prompt(text) == (3, "alice", "reflection")
    assert tag_prompt("no header here") == (None, None, None)


def test_turn_credit_by_type():
    assert turn_credit(REWARDS, 2, "alice", "verdict") == ("2|alice|verdict", 1.0)
    # a message decides what both agents can verify: mean of the two verdict rewards
    assert turn_credit(REWARDS, 2, "alice", "communication") == ("2|alice|message", 0.5)
    assert turn_credit(REWARDS, 1, "bob", "communication") == ("1|bob|message", -0.5)
    # a reflection is credited with the same agent's verdict in the next episode
    assert turn_credit(REWARDS, 1, "alice", "reflection") == ("1|alice|reflection", 1.0)
    assert turn_credit(REWARDS, 2, "bob", "reflection") == ("2|bob|reflection", -1.0)
    # the reflection of the last episode has no consequence inside the rollout
    assert turn_credit(REWARDS, 3, "alice", "reflection") == (DROP, None)
    # a task turn is credited with the agent's own task result
    assert turn_credit(REWARDS, 1, "bob", "task", TASK_OK) == ("1|bob|task", -1.0)
    assert turn_credit(REWARDS, 3, "bob", "task", TASK_OK) == ("3|bob|task", 1.0)


def test_turn_credit_unattributable_rows():
    assert turn_credit(None, 1, "alice", "verdict") == (None, None)       # rollout without a credit file
    assert turn_credit(REWARDS, None, "alice", "verdict") == (None, None)  # no episode header
    assert turn_credit(REWARDS, 4, "alice", "verdict") == (None, None)     # episode outside the rollout
    assert turn_credit(REWARDS, 1, None, "verdict") == (None, None)        # no role line
    assert turn_credit(REWARDS, 1, "alice", "task", None) == (None, None)  # no task results


def test_trained_types_switch():
    old = os.environ.get("PROVE_TASK_CREDIT")
    try:
        os.environ["PROVE_TASK_CREDIT"] = "1"
        assert "task" in trained_types()
        os.environ["PROVE_TASK_CREDIT"] = "0"
        assert trained_types() == ("communication", "verdict", "reflection")
    finally:
        os.environ.pop("PROVE_TASK_CREDIT", None)
        if old is not None:
            os.environ["PROVE_TASK_CREDIT"] = old


def test_response_shape():
    assert response_shape_ok("verdict", 'thinking</think>\n{"verdict": "reject"}', True)
    assert not response_shape_ok("verdict", 'thinking</think>\n{"verdict": "reject"}', False)  # truncated
    assert not response_shape_ok("verdict", "thinking</think>\nI reject.", True)
    assert response_shape_ok("reflection", "</think>I rejected because the log did not arrive.", True)
    assert not response_shape_ok("reflection", "</think><tool_call>", True)
    assert response_shape_ok("task", '</think>[{"name": "run_tests"}]', True)
    assert response_shape_ok("task", "</think><tool_call>\n<function=run_tests>", True)
    assert not response_shape_ok("task", "</think>Let me think about it.", True)


def test_turn_advantages_compare_the_same_turn_across_rollouts():
    # One group of four rollouts. Verdict turn of alice in episode 1: rewards 0, 0, -1, -1.
    # Rollout r1 has two rows for the same message turn; they share one value and one advantage.
    groups = ["g"] * 6
    rollouts = ["r1", "r2", "r3", "r4", "r1", "r1"]
    keys = ["1|alice|verdict"] * 4 + ["1|alice|message"] * 2
    rewards = [0.0, 0.0, -1.0, -1.0, 0.5, 0.5]
    adv, info = turn_advantages(groups, rollouts, keys, rewards)
    std = math.sqrt(((0.5 ** 2) * 4) / 3)
    assert [round(a, 4) for a in adv[:4]] == [round(0.5 / std, 4)] * 2 + [round(-0.5 / std, 4)] * 2
    assert adv[4] == adv[5] == 0.0  # a turn seen in one rollout only has nothing to be compared with
    assert info["prove/turns"] == 2 and info["prove/degenerate_turn_frac"] == 0.5


def test_turn_advantages_groups_and_untagged_rows():
    groups = ["g1", "g1", "g2", "g2", "g1"]
    rollouts = ["a", "b", "c", "d", "a"]
    keys = ["1|bob|verdict"] * 4 + [None]
    rewards = [1.0, -1.0, 0.0, 0.0, None]
    adv, info = turn_advantages(groups, rollouts, keys, rewards)
    assert adv[0] > 0 > adv[1] and abs(adv[0] + adv[1]) < 1e-9
    assert adv[2] == adv[3] == 0.0   # identical rewards in the group
    assert adv[4] is None            # untagged row keeps the advantage it had
    assert info["prove/rows_tagged_frac"] == 0.8


def test_credit_file_round_trip(tmp_path):
    outcomes = [{"reward": r, "task_ok": t} for r, t in zip(REWARDS, TASK_OK)]
    write_credit(str(tmp_path), "ro-123", outcomes, {"seq": 7})
    assert load_credit(str(tmp_path), "ro-123") == (REWARDS, TASK_OK)
    assert load_credit(str(tmp_path), "missing") == (None, None)
    assert rollout_id_from_event_url("http://localhost:8181/v1/rollouts/ro-123/attempts/at-1/events") == "ro-123"
