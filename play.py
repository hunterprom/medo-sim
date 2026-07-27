"""Run a trained policy in the interactive MuJoCo viewer.

    python play.py runs/first_walk/ckpt_latest.pt

Keys (click the viewer window first):
    UP / DOWN     forward speed +/- 0.1 m/s
    LEFT / RIGHT  turn rate +/- 0.1 rad/s
    A / D         sideways speed +/- 0.1 m/s
    SPACE         zero the command
    BACKSPACE     reset the robot
Drag with right mouse + Ctrl to shove the robot (MuJoCo perturbation).
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from medo_sim.env import DEFAULT_MODEL, EnvConfig, MedoEnv, layout_for_n_obs
from medo_sim.ppo import PPO

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_SPACE, KEY_BACKSPACE = 32, 259
KEY_A, KEY_D = 65, 68


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cmd", nargs=3, type=float, default=None,
                   metavar=("VX", "VY", "WZ"),
                   help="start with this command active (keys still adjust)")
    args = p.parse_args()

    n_in = PPO.peek_n_obs(args.checkpoint)
    cfg = EnvConfig(model_path=args.model, randomize=False, obs_noise=False,
                    **layout_for_n_obs(n_in))
    env = MedoEnv(cfg, seed=0)
    agent = PPO(env.n_obs, env.nu, device=args.device)
    agent.load(args.checkpoint)
    print(f"loaded {args.checkpoint} ({n_in} obs)")
    print(__doc__)

    # straight-line specialists (heading-error obs) were trained on
    # forward/backward only: cap keys to their envelope. Turning still works
    # gently -- LEFT/RIGHT rotate the commanded heading and the policy
    # steers to follow -- but sideways was never trained.
    if cfg.obs_heading_err:
        vx_hi, vx_lo, vy_cap, wz_cap = 0.5, -0.4, 0.0, 0.3
        print("straight-line policy: fwd/back capped 0.5/-0.4, no strafing, "
              "gentle turns only")
    else:
        vx_hi, vx_lo, vy_cap, wz_cap = 0.7, -0.5, 0.35, 0.9

    command = (np.zeros(3) if args.cmd is None
               else np.array(args.cmd, dtype=float))
    reset_flag = [False]

    def on_key(keycode):
        # caps match the checkpoint's trained command ranges: beyond them the
        # policy is being asked for speeds it never trained on
        if keycode == KEY_UP:
            command[0] = min(command[0] + 0.1, vx_hi)
        elif keycode == KEY_DOWN:
            command[0] = max(command[0] - 0.1, vx_lo)
        elif keycode == KEY_LEFT:
            command[2] = min(command[2] + 0.1, wz_cap)
        elif keycode == KEY_RIGHT:
            command[2] = max(command[2] - 0.1, -wz_cap)
        elif keycode == KEY_A:
            command[1] = min(command[1] + 0.1, vy_cap)
        elif keycode == KEY_D:
            command[1] = max(command[1] - 0.1, -vy_cap)
        elif keycode == KEY_SPACE:
            command[:] = 0.0
        elif keycode == KEY_BACKSPACE:
            reset_flag[0] = True
        print(f"cmd: vx {command[0]:+.1f}  vy {command[1]:+.1f}  wz {command[2]:+.1f}")

    obs = env.reset()
    env.command = command.copy()

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as v:
        while v.is_running():
            step_start = time.time()
            if reset_flag[0]:
                obs = env.reset()
                reset_flag[0] = False
            env.command = command.copy()
            action = agent.act_deterministic(obs)
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                obs = env.reset()
            v.sync()
            # real-time pacing
            leftover = env.dt - (time.time() - step_start)
            if leftover > 0:
                time.sleep(leftover)


if __name__ == "__main__":
    main()
