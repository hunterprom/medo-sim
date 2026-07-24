r"""Behavior-clone a scripted trot into a policy checkpoint as a warm start.

Why: PPO from scratch (and from a marching plateau) failed to discover
translation, while a hand-scripted firmware-style trot walks the model at
~0.09 m/s. So we teach the actor that gait directly, then let RL polish it.

    python tools/bc_warmstart.py runs\walk3_deploy\ckpt_latest.pt ^
        runs\walk3_deploy\ckpt_bc.pt

Generates demonstrations with the scripted trot at several commanded
speeds (stride scales with speed; standing holds home), regresses the
actor onto them (obs normalizer and critic untouched), and saves a
checkpoint you can pass to train.py --resume.
"""

from __future__ import annotations

import os
import sys

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medo_sim.env import EnvConfig, MedoEnv  # noqa: E402
from medo_sim.ppo import PPO, PPOConfig  # noqa: E402

LEGS = ["fr", "fl", "br", "bl"]
PAIR_A = ["fr", "bl"]
STEP_H = 0.022             # swing lift; the scripted-trot experiment walked
CYCLE_S = 1.0              # must equal 1/EnvConfig.gait_clock_hz
N_PHASE = 20
MAX_STRIDE = 0.06          # firmware stride; slight action clipping accepted
# demo commands: vx and the stride that produces it (stride = vx * cycle)
DEMO_VX = [-0.25, -0.12, 0.0, 0.09, 0.17, 0.25]
STEPS_PER_DEMO = 1200      # 24 s each at 50 Hz
TEACHER_NOISE = 0.08       # noise on APPLIED actions; labels stay clean, so
                           # every sample teaches recovery toward the gait
BC_EPOCHS = 60
BC_LR = 1e-3


class TrotScript:
    """Foot-target trot turned into ctrl vectors via FK lookup."""

    def __init__(self, model, home_qpos):
        self.model = model
        self.home_qpos = home_qpos
        self.data = mujoco.MjData(model)
        self.jadr = {}
        for leg in LEGS:
            for jn in ["thigh", "knee"]:
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{jn}")
                self.jadr[(leg, jn)] = model.jnt_qposadr[j]
        self.foot_gid = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                                f"{leg}_foot") for leg in LEGS}
        self.data.qpos[:] = home_qpos
        mujoco.mj_kinematics(model, self.data)
        self.home_foot = {leg: self.data.geom_xpos[self.foot_gid[leg]].copy()
                          for leg in LEGS}

    def _solve_leg(self, leg, dx, dz):
        target = self.home_foot[leg] + np.array([dx, 0.0, dz])

        def cost(a, b):
            self.data.qpos[:] = self.home_qpos
            self.data.qpos[self.jadr[(leg, "thigh")]] += a
            self.data.qpos[self.jadr[(leg, "knee")]] += b
            mujoco.mj_kinematics(self.model, self.data)
            e = self.data.geom_xpos[self.foot_gid[leg]] - target
            return e[0] * e[0] + 4.0 * e[2] * e[2] + 0.1 * e[1] * e[1]

        grid = np.linspace(-0.9, 0.9, 25)
        best = min(((cost(a, b), a, b) for a in grid for b in grid))
        step = 0.075
        for _ in range(4):
            _, a0, b0 = best
            best = min(((cost(a, b), a, b)
                        for a in np.linspace(a0 - step, a0 + step, 7)
                        for b in np.linspace(b0 - step, b0 + step, 7)))
            step /= 3.0
        return best[1], best[2]

    def build_lut(self, stride):
        """(N_PHASE, nu) ctrl table for one stride length (signed)."""
        lut = np.tile(self.home_qpos[7:].copy(), (N_PHASE, 1))
        if abs(stride) < 1e-6:
            return lut
        for p in range(N_PHASE):
            phase = p / N_PHASE
            half = (phase * 2.0) % 1.0
            a_swings = phase < 0.5
            ang = np.pi * half
            swing = (-(stride / 2) * np.cos(ang), STEP_H * np.sin(ang))
            stance = ((stride / 2) - stride * half, 0.0)
            for leg in LEGS:
                swinging = (leg in PAIR_A) == a_swings
                dx, dz = swing if swinging else stance
                da, db = self._solve_leg(leg, dx, dz)
                i = LEGS.index(leg) * 3
                lut[p, i + 1] += da
                lut[p, i + 2] += db
        return lut


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else r"runs\walk3_deploy\ckpt_latest.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else r"runs\walk3_deploy\ckpt_bc.pt"

    # target layout: deploy obs + gait clock. The source checkpoint may be
    # clockless (45 dims); its first-layer weights are grafted in with
    # zero-initialized columns for the two clock inputs.
    cfg = EnvConfig(randomize=False, obs_noise=False,
                    obs_lin_vel=False, obs_gait_clock=True)
    env = MedoEnv(cfg, seed=11)
    agent = PPO(env.n_obs, env.nu, PPOConfig(), device="cpu")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    net_sd = ckpt["net"]
    n_src = net_sd["actor.0.weight"].shape[1]
    if n_src == env.n_obs - 2:
        for key in ("actor.0.weight", "critic.0.weight"):
            w = net_sd[key]
            net_sd[key] = torch.cat([w, torch.zeros(w.shape[0], 2)], dim=1)
        rms = ckpt["obs_rms"]
        rms["mean"] = np.concatenate([rms["mean"], [0.0, 0.0]])
        rms["var"] = np.concatenate([rms["var"], [0.5, 0.5]])
        print(f"grafted {n_src}-dim checkpoint into {env.n_obs}-dim net")
    elif n_src != env.n_obs:
        raise SystemExit(f"checkpoint obs {n_src} incompatible with {env.n_obs}")
    agent.net.load_state_dict(net_sd)
    agent.obs_rms.load_state_dict(ckpt["obs_rms"])
    agent._sync_act_net()
    gstep = ckpt.get("global_step", 0)
    script = TrotScript(env.model, env.home_qpos)

    rng = np.random.default_rng(3)
    luts = {}
    for vx in DEMO_VX:
        stride = float(np.clip(vx * CYCLE_S, -MAX_STRIDE, MAX_STRIDE))
        luts[vx] = script.build_lut(stride)

    def script_action(vx):
        # keyed to the ENV's own gait clock, so the label is a true function
        # of the observation (which contains sin/cos of this exact phase)
        ctrl = luts[vx][int(env.gait_phase * N_PHASE) % N_PHASE]
        return (ctrl - env.default_joint_pos) / cfg.action_scale

    def collect(round_name, use_policy):
        """Roll out; label every visited state with the script's action."""
        obs_l, act_l = [], []
        for vx in DEMO_VX:
            obs = env.reset()
            env.command = np.array([vx, 0.0, 0.0])
            walked, resets = [], 0
            for s in range(STEPS_PER_DEMO):
                label = np.clip(script_action(vx), -1.0, 1.0)
                obs_l.append(obs)
                act_l.append(label)
                if use_policy:
                    applied = agent.act_deterministic(
                        obs.astype(np.float64, copy=False))
                    applied = np.clip(applied, -1.0, 1.0)
                else:
                    applied = np.clip(
                        label + rng.normal(0, TEACHER_NOISE, env.nu), -1.0, 1.0)
                obs, _, term, _, _ = env.step(applied)
                env.command = np.array([vx, 0.0, 0.0])
                walked.append(env.lin_vel[0])
                if term:
                    resets += 1
                    obs = env.reset()
                    env.command = np.array([vx, 0.0, 0.0])
            print(f"  {round_name} vx {vx:+.2f}: mean vx {np.mean(walked):+.3f}, "
                  f"{resets} falls")
        return obs_l, act_l

    def train_bc(X_np, Y_np):
        X = torch.from_numpy(agent.obs_rms.normalize(np.array(X_np, dtype=np.float64)))
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

    print("round 1: noisy teacher (script drives, labels clean)")
    obs1, act1 = collect("teacher", use_policy=False)
    train_bc(obs1, act1)

    print("round 2 (DAgger): policy drives, script labels its visited states")
    obs2, act2 = collect("policy", use_policy=True)
    train_bc(obs1 + obs2, act1 + act2)

    agent.save(dst, {"global_step": gstep})
    print(f"saved {dst} (global_step {gstep:,})")


if __name__ == "__main__":
    main()
