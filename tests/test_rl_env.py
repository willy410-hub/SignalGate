import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

from sgate_eval.annotation import krippendorff_alpha
from sgate_eval.rl import DataBudgetEnv, TrainConfig, evaluate, heuristic_policy, policy_from_net, random_policy, train
from sgate_eval.rl.env import N_ACTIONS, STOP, VENDOR, decode, encode, fast_alpha_binary, measure_batch, sample_batch


def test_fast_alpha_equals_reference_implementation():
    rng = np.random.default_rng(0)
    for acc, herd in [(0.95, False), (0.8, False), (0.65, False), (0.6, True)]:
        _, labels, _ = sample_batch(rng, acc, 150, herd)
        assert fast_alpha_binary(labels) == pytest.approx(krippendorff_alpha(labels.tolist(), "nominal"), abs=1e-9)


def test_action_encoding_roundtrip():
    for a in range(N_ACTIONS - 1):
        assert encode(*decode(a)) == a


def test_herding_vendor_has_high_agreement_but_low_gold_accuracy():
    rng = np.random.default_rng(1)
    alphas, golds = zip(*[measure_batch(rng, 0.6, 200, herd=True)[:2] for _ in range(30)])
    assert np.mean(alphas) > 0.7      # passes an agreement-only gate
    assert np.mean(golds) < 0.8       # fails the hidden-gold audit
    honest = [measure_batch(rng, 0.6, 200, herd=False)[0] for _ in range(30)]
    assert np.mean(honest) < 0.2


def test_budget_never_negative_and_episodes_terminate():
    env = DataBudgetEnv(family="train", seed=0)
    rng = np.random.default_rng(0)
    for _ in range(30):
        env.reset()
        done, n = False, 0
        while not done:
            feas = np.flatnonzero(env.action_mask())
            _, _, done, _, _ = env.step(int(rng.choice(feas)))
            assert env.budget >= -1e-9
            n += 1
        assert n <= env.cfg.max_steps


def test_infeasible_action_ends_episode_without_spending():
    env = DataBudgetEnv(seed=0)
    env.budget = 1.0
    before = env.ep_spend
    _, r, done, _, info = env.step(encode(2, 2, 2))
    assert done and r == 0.0 and env.ep_spend == before


def test_shield_masks_calibrated_bad_actions_and_vendor():
    env = DataBudgetEnv(shield=True, seed=0)
    env.reset()
    mask = env.action_mask()
    for a in range(N_ACTIONS - 1):
        d, t, _ = decode(a)
        if t == VENDOR:
            assert not mask[a]
        if env._calib[d, t] < env.alpha_min + env.shield_margin:
            assert not mask[a]
    assert mask[STOP]


def test_learning_signal_gain_is_monotone_in_label_quality():
    # expert labels in the weakest domain should help more than crowd labels, on average
    gains = {}
    for tier in (0, 2):
        env = DataBudgetEnv(seed=3, gold_audit=False)
        vals = []
        for _ in range(60):
            env.reset()
            _, _, _, _, info = env.step(encode(2, tier, 1))
            vals.append(info["ep_gain"])
        gains[tier] = np.mean(vals)
    assert gains[2] > gains[0]


def test_short_training_beats_random_on_true_gain():
    res = train(TrainConfig(method="unconstrained", total_steps=20_480, seed=0))
    learned = evaluate(policy_from_net(res.policy), "train", 100, seed=5)["ep_gain"].mean()
    rand = evaluate(random_policy(0), "train", 100, seed=5)["ep_gain"].mean()
    assert learned > rand


def test_heuristic_never_uses_vendor():
    r = evaluate(heuristic_policy(), "train", 50, seed=2)
    assert r["ep_vendor_steps"].sum() == 0
