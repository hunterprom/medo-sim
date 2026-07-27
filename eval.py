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

from medo_sim.env import (DEFAULT_MODEL, EnvConfig, MedoEnv, layout_for_n_obs,
                          yaw_of)
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

# for the straight-line specialist: forward/backward only, plus a fast probe
PROBES_STRAIGHT = [
    ("forward",      np.array([0.4, 0.0, 0.0])),
    ("forward_fast", np.array([0.6, 0.0, 0.0])),
    ("backward",     np.array([-0.4, 0.0, 0.0])),
    ("stand",        np.array([0.0, 0.0, 0.0])),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--episodes", type=int, default=3, help="episodes per probe")
    p.add_argument("--seconds", type=float, default=5.0, help="length of each episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probes", choices=["all", "straight"], default="all",
                   help="'straight' = forward/backward/stand only, with "
                        "lateral-drift and yaw-drift metrics")
    args = p.parse_args()
    probes = PROBES_STRAIGHT if args.probes == "straight" else PROBES

    n_in = PPO.peek_n_obs(args.checkpoint)
    cfg = EnvConfig(model_path=args.model, randomize=False, obs_noise=False,
                    **layout_for_n_obs(n_in))
    env = MedoEnv(cfg, seed=args.seed)
    agent = PPO(env.n_obs, env.nu, PPOConfig(), device="cpu")
    agent.load(args.checkpoint)
    steps = int(args.seconds / env.dt)

    print(f"model {args.model}")
    print(f"checkpoint {args.checkpoint}")
    print(f"{args.episodes} episodes x {args.seconds:.0f} s per probe\n")
    print(f"{'probe':<11} {'command':>19} {'achieved (mean)':>22}  {'err':>5}  falls")

    worst = 0.0
    for name, cmd in probes:
        vels, falls, drifts = [], 0, []
        for ep in range(args.episodes):
            env.rng = np.random.default_rng(args.seed + 100 * ep)
            obs = env.reset()
            env.command = cmd.copy()
            p0 = env.data.qpos[0:2].copy()
            yaw0 = yaw_of(env.data.qpos[3:7])
            for _ in range(steps):
                action = agent.act_deterministic(obs)
                obs, _, term, trunc, _ = env.step(action)
                env.command = cmd.copy()  # pin the probe command
                vels.append([env.lin_vel[0], env.lin_vel[1], env.ang_vel[2]])
                if term:
                    falls += 1
                    break
            # straightness: lateral offset and yaw change per meter traveled
            d = env.data.qpos[0:2] - p0
            dist = float(np.hypot(*d))
            if abs(cmd[0]) > 0.05 and cmd[2] == 0.0 and dist > 0.05:
                lat = float(-np.sin(yaw0) * d[0] + np.cos(yaw0) * d[1])
                dyaw = (yaw_of(env.data.qpos[3:7]) - yaw0 + np.pi) \
                    % (2.0 * np.pi) - np.pi
                drifts.append([100.0 * abs(lat) / dist,
                               np.degrees(abs(dyaw)) / dist])
        mean = np.mean(vels, axis=0)
        err = float(np.linalg.norm(mean - cmd))
        worst = max(worst, err)
        extra = ""
        if drifts:
            md = np.mean(drifts, axis=0)
            extra = f"   drift {md[0]:5.1f} cm/m, {md[1]:5.1f} deg/m"
        print(f"{name:<12} "
              f"[{cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f}] "
              f"   [{mean[0]:+.2f} {mean[1]:+.2f} {mean[2]:+.2f}] "
              f" {err:5.2f}  {falls}/{args.episodes}{extra}")

    print(f"\nworst tracking error: {worst:.2f} "
          f"(under ~0.15 = solid, under ~0.3 = usable)")
    print("straight walk targets: drift under ~5 cm/m and ~5 deg/m")


if __name__ == "__main__":
    main()
