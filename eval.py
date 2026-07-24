r"""Evaluate a trained policy on fixed direction probes, no viewer needed.

    python eval.py runs\walk1\ckpt_latest.pt

Runs the policy for a few episodes per probe (forward, backward, strafe
left/right, turn both ways, stand) with domain randomization and noise off,
and reports commanded vs achieved velocity plus falls. This is the pass/fail
check that the dog actually walks in every direction.
"""

from __future__ import annotations

import argparse

import numpy as np

from medo_sim.env import DEFAULT_MODEL, EnvConfig, MedoEnv
from medo_sim.ppo import PPO, PPOConfig

PROBES = [
    ("forward",    np.array([0.4, 0.0, 0.0])),
    ("backward",   np.array([-0.4, 0.0, 0.0])),
    ("left",       np.array([0.0, 0.3, 0.0])),
    ("right",      np.array([0.0, -0.3, 0.0])),
    ("turn_left",  np.array([0.0, 0.0, 0.8])),
    ("turn_right", np.array([0.0, 0.0, -0.8])),
    ("stand",      np.array([0.0, 0.0, 0.0])),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--episodes", type=int, default=3, help="episodes per probe")
    p.add_argument("--seconds", type=float, default=5.0, help="length of each episode")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    n_in = PPO.peek_n_obs(args.checkpoint)
    cfg = EnvConfig(model_path=args.model, randomize=False, obs_noise=False)
    env = MedoEnv(cfg, seed=args.seed)
    if n_in == env.n_obs - 3:  # checkpoint trained with --deploy (no lin vel)
        cfg = EnvConfig(model_path=args.model, randomize=False,
                        obs_noise=False, obs_lin_vel=False)
        env = MedoEnv(cfg, seed=args.seed)
    if n_in != env.n_obs:
        raise SystemExit(f"checkpoint expects {n_in} obs, model gives {env.n_obs}")
    agent = PPO(env.n_obs, env.nu, PPOConfig(), device="cpu")
    agent.load(args.checkpoint)
    steps = int(args.seconds / env.dt)

    print(f"model {args.model}")
    print(f"checkpoint {args.checkpoint}")
    print(f"{args.episodes} episodes x {args.seconds:.0f} s per probe\n")
    print(f"{'probe':<11} {'command':>19} {'achieved (mean)':>22}  {'err':>5}  falls")

    worst = 0.0
    for name, cmd in PROBES:
        vels, falls = [], 0
        for ep in range(args.episodes):
            env.rng = np.random.default_rng(args.seed + 100 * ep)
            obs = env.reset()
            env.command = cmd.copy()
            for _ in range(steps):
                action = agent.act_deterministic(obs)
                obs, _, term, trunc, _ = env.step(action)
                env.command = cmd.copy()  # pin the probe command
                vels.append([env.lin_vel[0], env.lin_vel[1], env.ang_vel[2]])
                if term:
                    falls += 1
                    break
        mean = np.mean(vels, axis=0)
        err = float(np.linalg.norm(mean - cmd))
        worst = max(worst, err)
        print(f"{name:<11} "
              f"[{cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f}] "
              f"   [{mean[0]:+.2f} {mean[1]:+.2f} {mean[2]:+.2f}] "
              f" {err:5.2f}  {falls}/{args.episodes}")

    print(f"\nworst tracking error: {worst:.2f} "
          f"(under ~0.15 = solid, under ~0.3 = usable)")


if __name__ == "__main__":
    main()
