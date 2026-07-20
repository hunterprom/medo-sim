"""Self-contained PPO for the Medo locomotion task.

Actor-critic MLPs with a learned state-independent log-std, GAE, clipped
surrogate objective, and legged-gym-style adaptive learning rate driven by the
analytic KL between the old and new Gaussian policies. Observations are
normalized by a running mean/std that ships inside checkpoints.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn


@dataclasses.dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    num_minibatches: int = 2
    ent_coef: float = 0.004
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    desired_kl: float = 0.01
    init_log_std: float = -0.2
    hidden: tuple = (256, 128)


class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


def mlp(sizes, act=nn.ELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, n_obs: int, n_act: int, cfg: PPOConfig):
        super().__init__()
        self.actor = mlp([n_obs, *cfg.hidden, n_act])
        self.critic = mlp([n_obs, *cfg.hidden, 1])
        self.log_std = nn.Parameter(torch.full((n_act,), cfg.init_log_std))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class PPO:
    def __init__(self, n_obs: int, n_act: int, cfg: PPOConfig | None = None,
                 device: str = "cpu"):
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device)
        self.net = ActorCritic(n_obs, n_act, self.cfg).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.obs_rms = RunningMeanStd((n_obs,))
        self.lr = self.cfg.lr

    @torch.no_grad()
    def act(self, obs_raw: np.ndarray, update_rms: bool = True):
        """Sample actions. Returns (action, logprob, value, mu, sigma) tensors."""
        if update_rms:
            self.obs_rms.update(obs_raw)
        obs = torch.from_numpy(self.obs_rms.normalize(obs_raw)).to(self.device)
        dist = self.net.dist(obs)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        value = self.net.value(obs)
        return obs, action, logprob, value, dist.loc, dist.scale

    @torch.no_grad()
    def act_deterministic(self, obs_raw: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(self.obs_rms.normalize(obs_raw[None])).to(self.device)
        return self.net.actor(obs)[0].cpu().numpy()

    @torch.no_grad()
    def value_of(self, obs_raw: np.ndarray) -> torch.Tensor:
        obs = torch.from_numpy(self.obs_rms.normalize(obs_raw)).to(self.device)
        return self.net.value(obs)

    def update(self, batch: dict) -> dict:
        c = self.cfg
        n = batch["obs"].shape[0]
        idx = torch.randperm(n, device=self.device)
        mb_size = n // c.num_minibatches
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        count = 0

        for _ in range(c.epochs):
            for start in range(0, n, mb_size):
                mb = idx[start:start + mb_size]
                obs = batch["obs"][mb]
                dist = self.net.dist(obs)
                logprob = dist.log_prob(batch["actions"][mb]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                value = self.net.value(obs)

                # analytic KL(old || new) between diagonal Gaussians -> adapt lr
                with torch.no_grad():
                    mu0, sd0 = batch["mu"][mb], batch["sigma"][mb]
                    kl = (torch.log(dist.scale / sd0)
                          + (sd0**2 + (mu0 - dist.loc) ** 2) / (2.0 * dist.scale**2)
                          - 0.5).sum(-1).mean()
                    if c.desired_kl > 0:
                        if kl > c.desired_kl * 2.0:
                            self.lr = max(1e-5, self.lr / 1.5)
                        elif kl < c.desired_kl / 2.0 and kl > 0.0:
                            self.lr = min(1e-2, self.lr * 1.5)
                        for g in self.opt.param_groups:
                            g["lr"] = self.lr

                ratio = (logprob - batch["logprobs"][mb]).exp()
                adv = batch["advantages"][mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pi_loss = -torch.min(
                    ratio * adv,
                    ratio.clamp(1.0 - c.clip_eps, 1.0 + c.clip_eps) * adv,
                ).mean()
                v_loss = (value - batch["returns"][mb]).pow(2).mean()
                loss = pi_loss + c.vf_coef * v_loss - c.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()

                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += entropy.item()
                stats["kl"] += kl.item()
                count += 1

        return {k: v / count for k, v in stats.items()} | {"lr": self.lr}

    # --------------------------------------------------------- persistence

    def save(self, path: str, extra: dict | None = None):
        torch.save({
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "lr": self.lr,
            **(extra or {}),
        }, path)

    def load(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        self.opt.load_state_dict(ckpt["opt"])
        self.obs_rms.load_state_dict(ckpt["obs_rms"])
        self.lr = ckpt.get("lr", self.cfg.lr)
        for g in self.opt.param_groups:
            g["lr"] = self.lr
        return ckpt
