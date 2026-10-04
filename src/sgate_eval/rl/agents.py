"""Masked PPO (CPU), a Lagrangian-constrained variant, and simple baselines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .env import (
    BATCH_SIZES, COST_PER_ITEM, D, DOMAIN_WEIGHT, N_ACTIONS, STOP, T, VENDOR, DataBudgetEnv, encode,
)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        h = self.body(obs)
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits), self.v(h).squeeze(-1)


@dataclass
class TrainConfig:
    method: str = "unconstrained"      # 'unconstrained' | 'shield' | 'lagrangian'
    reward_mode: str = "true_gain"     # 'true_gain' | 'agreement'
    gold_audit: bool = True
    total_steps: int = 60_000
    rollout: int = 2048
    epochs: int = 4
    minibatch: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent: float = 0.01
    violation_limit: float = 0.5       # expected violations per episode (CMDP budget)
    lagrange_lr: float = 0.2
    lagrange_init: float = 1.0
    seed: int = 0


@dataclass
class TrainResult:
    policy: ActorCritic
    cfg: TrainConfig
    lagrange: float
    history: List[Dict[str, float]] = field(default_factory=list)


def _env_for(cfg: TrainConfig, family: str, seed: int) -> DataBudgetEnv:
    return DataBudgetEnv(
        family=family, reward_mode=cfg.reward_mode, gold_audit=cfg.gold_audit,
        shield=(cfg.method == "shield"), seed=seed,
    )


def train(cfg: TrainConfig) -> TrainResult:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    env = _env_for(cfg, "train", cfg.seed)
    net = ActorCritic(env.observation_space.shape[0], N_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    lam_mult = cfg.lagrange_init if cfg.method == "lagrangian" else 0.0
    obs, _ = env.reset(seed=cfg.seed)
    hist: List[Dict[str, float]] = []
    steps = 0
    while steps < cfg.total_steps:
        O, A, LP, R, V, M, DN = [], [], [], [], [], [], []
        ep_viol, ep_gain, ep_proxy = [], [], []
        for _ in range(cfg.rollout):
            mask = env.action_mask()
            with torch.no_grad():
                dist, v = net(torch.as_tensor(obs).unsqueeze(0), torch.as_tensor(mask).unsqueeze(0))
                a = dist.sample()
            nobs, r, done, _, info = env.step(int(a))
            r_adj = r - lam_mult * info["violation"]
            O.append(obs); A.append(int(a)); LP.append(float(dist.log_prob(a))); R.append(r_adj)
            V.append(float(v)); M.append(mask); DN.append(done)
            if done:
                ep_viol.append(info["ep_violations"]); ep_gain.append(info["ep_gain"]); ep_proxy.append(info["ep_proxy"])
                nobs, _ = env.reset()
            obs = nobs
        steps += cfg.rollout
        with torch.no_grad():
            _, last_v = net(torch.as_tensor(obs).unsqueeze(0), torch.as_tensor(env.action_mask()).unsqueeze(0))
        adv = np.zeros(cfg.rollout, dtype=np.float32)
        gae, next_v = 0.0, float(last_v)
        for i in reversed(range(cfg.rollout)):
            nonterm = 0.0 if DN[i] else 1.0
            delta = R[i] + cfg.gamma * next_v * nonterm - V[i]
            gae = delta + cfg.gamma * cfg.lam * nonterm * gae
            adv[i] = gae
            next_v = V[i]
        ret = adv + np.array(V, dtype=np.float32)
        t_obs, t_act = torch.as_tensor(np.array(O)), torch.as_tensor(A)
        t_lp, t_adv, t_ret = torch.as_tensor(LP, dtype=torch.float32), torch.as_tensor(adv), torch.as_tensor(ret)
        t_mask = torch.as_tensor(np.array(M))
        t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)
        for _ in range(cfg.epochs):
            perm = torch.randperm(cfg.rollout)
            for s in range(0, cfg.rollout, cfg.minibatch):
                idx = perm[s : s + cfg.minibatch]
                dist, v = net(t_obs[idx], t_mask[idx])
                ratio = torch.exp(dist.log_prob(t_act[idx]) - t_lp[idx])
                pg = -torch.min(ratio * t_adv[idx], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * t_adv[idx]).mean()
                loss = pg + 0.5 * (v - t_ret[idx]).pow(2).mean() - cfg.ent * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        if cfg.method == "lagrangian" and ep_viol:
            lam_mult = max(0.0, lam_mult + cfg.lagrange_lr * (float(np.mean(ep_viol)) - cfg.violation_limit))
        hist.append({
            "steps": steps, "ep_gain": float(np.mean(ep_gain)) if ep_gain else float("nan"),
            "ep_violations": float(np.mean(ep_viol)) if ep_viol else float("nan"),
            "ep_proxy": float(np.mean(ep_proxy)) if ep_proxy else float("nan"), "lagrange": lam_mult,
        })
    return TrainResult(net, cfg, lam_mult, hist)


# ---------------------------------------------------------------- policies
def policy_from_net(net: ActorCritic, greedy: bool = False) -> Callable[[DataBudgetEnv, np.ndarray], int]:
    def act(env: DataBudgetEnv, obs: np.ndarray) -> int:
        with torch.no_grad():
            dist, _ = net(torch.as_tensor(obs).unsqueeze(0), torch.as_tensor(env.action_mask()).unsqueeze(0))
            return int(dist.probs.argmax()) if greedy else int(dist.sample())
    return act


def random_policy(seed: int = 0) -> Callable[[DataBudgetEnv, np.ndarray], int]:
    rng = np.random.default_rng(seed)
    def act(env, obs):
        feas = np.flatnonzero(env.action_mask()[:STOP])
        return int(rng.choice(feas)) if feas.size else STOP
    return act


def heuristic_policy(alpha_floor: float = 0.75) -> Callable[[DataBudgetEnv, np.ndarray], int]:
    """Rule of thumb: rotate domains by weight; in each use the cheapest tier whose
    calibration alpha clears the floor (never the bulk vendor); batch of 100."""
    order = [2, 1, 2, 0, 1, 2]
    state = {"i": 0}
    def act(env, obs):
        if env.t == 0:
            state["i"] = 0
        mask = env.action_mask()
        for _ in range(len(order)):
            d = order[state["i"] % len(order)]
            state["i"] += 1
            tiers = [t for t in np.argsort(COST_PER_ITEM) if t != VENDOR and env._calib[d, t] >= alpha_floor]
            for t in tiers:
                a = encode(d, int(t), 1)
                if mask[a]:
                    return a
        return STOP
    return act


def evaluate(policy_fn, family: str, n_episodes: int = 200, seed: int = 999, **env_kw) -> Dict[str, np.ndarray]:
    env = DataBudgetEnv(family=family, seed=seed, **env_kw)
    out = {k: [] for k in ("ep_gain", "ep_violations", "ep_spend_frac", "ep_vendor_steps", "ep_proxy")}
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy_fn(env, obs))
        for k in out:
            out[k].append(info[k])
    return {k: np.array(v) for k, v in out.items()}
