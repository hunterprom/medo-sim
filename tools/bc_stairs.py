r"""Behavior-clone a HIGH-STEPPING trot on stair terrain as a warm start.

RL alone plateaued at "two steps then fall": sustained blind stair
climbing was not being discovered. The classic blind-climber gait is a
trot with exaggerated foot lift -- it clears step nosings without seeing
them. Clone it on a mix of stairs and flat episodes (one gait for both:
a blind policy cannot switch before contact), then let RL polish.

    python tools/bc_stairs.py runs\walk11_terrain\ckpt_latest.pt ^
        runs\walk11_terrain\ckpt_bc_stairs.pt
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medo_sim.env import EnvConfig, MedoEnv  # noqa: E402
from medo_sim.ppo import PPO, PPOConfig  # noqa: E402
from tools.bc_warmstart import TrotScript, N_PHASE  # noqa: E402

import tools.bc_warmstart as bw  # noqa: E402

bw.STEP_H = 0.055          # high knees: swing apex ~5.5 cm clears 4-5 cm steps
STEP_H = bw.STEP_H

DEMO = [                    # (vx, stairs difficulty or None for flat)
    (0.15, None), (0.25, None),
    (0.15, 0.1), (0.15, 0.25), (0.18, 0.4), (0.15, 0.55),
]
STEPS_PER_DEMO = 1000
TEACHER_NOISE = 0.08
BC_EPOCHS = 50
BC_LR = 1e-3


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    cfg = EnvConfig(randomize=False, obs_noise=False,
                    obs_lin_vel=False, obs_gait_clock=True,
                    obs_heading_err=True, terrain=True,
                    terrain_kinds=("stairs",), terrain_flat_prob=0.0,
                    terrain_diff_up=0.0, terrain_diff_down=0.0)
    env = MedoEnv(cfg, seed=23)
    agent = PPO(env.n_obs, env.nu, PPOConfig(), device="cpu")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if ckpt["net"]["actor.0.weight"].shape[1] != env.n_obs:
        raise SystemExit("layout mismatch")
    agent.net.load_state_dict(ckpt["net"])
    agent.obs_rms.load_state_dict(ckpt["obs_rms"])
    agent._sync_act_net()
    gstep = ckpt.get("global_step", 0)
    script = TrotScript(env.model, env.home_qpos)

    rng = np.random.default_rng(9)
    luts = {}
    for vx, _ in DEMO:
        stride = float(np.clip(vx * 1.0, -0.08, 0.08))
        if vx not in luts:
            luts[vx] = script.build_lut(stride)

    def script_action(vx):
        ctrl = luts[vx][int(env.gait_phase * N_PHASE) % N_PHASE]
        return np.clip((ctrl - env.default_joint_pos) / cfg.action_scale,
                       -1.0, 1.0)

    def set_terrain(diff):
        if diff is None:
            env.cfg.terrain_flat_prob = 1.0
        else:
            env.cfg.terrain_flat_prob = 0.0
            env.difficulty = diff

    def collect(round_name, use_policy):
        obs_l, act_l = [], []
        for vx, diff in DEMO:
            set_terrain(diff)
            obs = env.reset()
            env.command = np.array([vx, 0.0, 0.0])
            prog, falls = [], 0
            for s in range(STEPS_PER_DEMO):
                label = script_action(vx)
                obs_l.append(obs)
                act_l.append(label)
                if use_policy:
                    applied = np.clip(agent.act_deterministic(
                        obs.astype(np.float64, copy=False)), -1.0, 1.0)
                else:
                    applied = np.clip(
                        label + rng.normal(0, TEACHER_NOISE, env.nu), -1, 1)
                obs, _, term, _, _ = env.step(applied)
                env.command = np.array([vx, 0.0, 0.0])
                if term:
                    falls += 1
                    obs = env.reset()
                    env.command = np.array([vx, 0.0, 0.0])
            d = "flat" if diff is None else f"stairs d={diff}"
            print(f"  {round_name} vx {vx:+.2f} {d}: x {env.data.qpos[0]:+.2f} "
                  f"z {env.data.qpos[2]:+.2f}, {falls} falls")
        return obs_l, act_l

    def train_bc(X_np, Y_np):
        X = torch.from_numpy(agent.obs_rms.normalize(
            np.array(X_np, dtype=np.float64)))
        Y = torch.from_numpy(np.array(Y_np, dtype=np.float32))
        opt = torch.optim.Adam(agent.net.actor.parameters(), lr=BC_LR)
        n = X.shape[0]
        for ep in range(BC_EPOCHS):
            idx = torch.randperm(n)
            tot = 0.0
            for s in range(0, n, 1024):
                mb = idx[s:s + 1024]
                loss = ((agent.net.actor(X[mb]) - Y[mb]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(mb)
            if ep % 20 == 0 or ep == BC_EPOCHS - 1:
                print(f"  bc epoch {ep:3d} mse {tot/n:.4f}")
        agent._sync_act_net()

    print("round 1: noisy high-step teacher")
    o1, a1 = collect("teacher", use_policy=False)
    train_bc(o1, a1)
    print("round 2 (DAgger): policy drives, teacher labels")
    o2, a2 = collect("policy", use_policy=True)
    train_bc(o1 + o2, a1 + a2)

    with torch.no_grad():
        agent.net.log_std.clamp_(max=-1.2)
    agent.save(dst, {"global_step": gstep})
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
