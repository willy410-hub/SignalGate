"""Reproducible RL experiments. Run: python -m sgate_eval.rl.experiments --out results/rl_results.json

E1  constraint handling: unconstrained PPO vs rule-based shield vs Lagrangian CMDP
    (plus random and heuristic baselines), evaluated on the training family and on
    a held-out family with shifted annotator accuracy.
E2  reward hacking: the same learner trained on an agreement-based proxy reward
    versus the true-gain reward, with the hidden-gold audit switched off in training,
    then evaluated with the audit on.
Every number comes from the simulator in env.py.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict

import numpy as np

from .agents import TrainConfig, evaluate, heuristic_policy, policy_from_net, random_policy, train

STEPS = 81_920
SEEDS = list(range(5))
N_EVAL = 200
CACHE_DIR = "results/rl_runs"


def _job(args) -> dict:
    exp, name, method, reward_mode, gold_train, seed = args
    cache = Path(CACHE_DIR) / f"{exp}_{name}_{seed}.json"
    if cache.exists():  # resumable: finished runs are never recomputed
        return json.loads(cache.read_text())
    out = _train_and_eval(args)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def _train_and_eval(args) -> dict:
    import torch
    torch.set_num_threads(1)
    exp, name, method, reward_mode, gold_train, seed = args
    cfg = TrainConfig(
        method=method, reward_mode=reward_mode, gold_audit=gold_train, total_steps=STEPS, seed=seed,
        lagrange_lr=0.05, lagrange_init=0.5,
    )
    res = train(cfg)
    pol = policy_from_net(res.policy)
    out = {"exp": exp, "name": name, "seed": seed, "history": res.history, "eval": {}}
    for fam in ("train", "heldout"):
        r = evaluate(
            pol, fam, N_EVAL, seed=1000 + seed,
            shield=(method == "shield"), gold_audit=(True if exp == "E2" else gold_train), reward_mode=reward_mode,
        )
        out["eval"][fam] = {k: float(v.mean()) for k, v in r.items()}
    return out


def run(out_path: str, workers: int) -> None:
    jobs = []
    for s in SEEDS:
        jobs += [("E1", "unconstrained", "unconstrained", "true_gain", True, s),
                 ("E1", "shield", "shield", "true_gain", True, s),
                 ("E1", "lagrangian", "lagrangian", "true_gain", True, s)]
        jobs += [("E2", "proxy_agreement", "unconstrained", "agreement", False, s),
                 ("E2", "true_gain", "unconstrained", "true_gain", False, s)]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_job, jobs), 1):
            results.append(r)
            print(f"[{i}/{len(jobs)}] {r['exp']} {r['name']} seed={r['seed']} heldout={r['eval']['heldout']}", flush=True)

    baselines = {}
    for name, pol in (("random", random_policy(0)), ("heuristic", heuristic_policy())):
        baselines[name] = {
            fam: {k: float(v.mean()) for k, v in evaluate(pol, fam, 1000, seed=4242).items()} for fam in ("train", "heldout")
        }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(
        {"steps": STEPS, "seeds": SEEDS, "n_eval": N_EVAL, "runs": results, "baselines": baselines}, indent=1))
    print("saved", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rl_results.json")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    run(a.out, a.workers)
