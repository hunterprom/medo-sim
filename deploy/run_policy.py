r"""Stream a trained policy to the physical Medo over the serial bridge.

    python deploy\run_policy.py runs\walk1\ckpt_latest.pt --port COM5
    python deploy\run_policy.py runs\walk1\ckpt_latest.pt --bench

BENCH-TEST ONLY for now: the dog has no IMU yet, so the observation is
assembled open-loop (gravity assumed straight down, zero base velocities,
joint state taken from the commanded targets). That is fine for checking
that the hardware moves coherently WITH THE DOG ON A STAND, but it is NOT
closed-loop walking -- do not put it on the floor and expect the sim gait.
Closed loop needs an IMU (MPU6050 on the same I2C bus) and a policy trained
without the base-linear-velocity observation; see deploy/README.md.

Keys while running: w/s forward speed, a/d strafe, q/e turn, space stop,
x exit (eases back to home).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402

from deploy import servo_map  # noqa: E402
from medo_sim.env import DEFAULT_MODEL, EnvConfig  # noqa: E402
from medo_sim.ppo import PPO, PPOConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--port", default=None, help="serial port, e.g. COM5")
    p.add_argument("--bench", action="store_true",
                   help="no serial: print the servo frames instead")
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--action-scale", type=float, default=None,
                   help="override EnvConfig.action_scale (start small, e.g. 0.25)")
    args = p.parse_args()
    if not args.bench and not args.port:
        p.error("--port COMx or --bench required")

    cfg = EnvConfig(model_path=args.model)
    scale = args.action_scale if args.action_scale is not None else cfg.action_scale
    model = mujoco.MjModel.from_xml_path(args.model)
    home = servo_map.home_from_model(model)
    nu = len(home)
    jnt_lo = model.jnt_range[1:, 0].copy()
    jnt_hi = model.jnt_range[1:, 1].copy()

    n_obs = 3 + 3 + 3 + nu + nu + nu + 3
    agent = PPO(n_obs, nu, PPOConfig(), device="cpu")
    agent.load(args.checkpoint)
    print(f"loaded {args.checkpoint}")
    print(__doc__)

    ser = None
    if not args.bench:
        import serial  # pyserial; pip install pyserial
        ser = serial.Serial(args.port, 115200, timeout=0.05)
        time.sleep(2.0)  # ESP32 reboots on port open
        ser.reset_input_buffer()
        ser.write(b"P\n")
        print("bridge:", ser.readline().decode(errors="replace").strip() or "(no reply)")

    try:
        import msvcrt  # windows console keys
        def poll_key():
            return msvcrt.getwch() if msvcrt.kbhit() else None
    except ImportError:
        def poll_key():
            return None

    dt = 1.0 / args.hz
    command = np.zeros(3)
    joint_pos = home.copy()
    prev_joint = home.copy()
    prev_action = np.zeros(nu)
    gravity = np.array([0.0, 0.0, -1.0])   # no IMU: assume level
    zeros3 = np.zeros(3)

    print("streaming at %.0f Hz; keys w/s a/d q/e space x" % args.hz)
    try:
        while True:
            t0 = time.time()
            k = poll_key()
            if k == "x":
                break
            elif k == "w":
                command[0] = min(command[0] + 0.1, 0.7)
            elif k == "s":
                command[0] = max(command[0] - 0.1, -0.5)
            elif k == "a":
                command[1] = min(command[1] + 0.1, 0.35)
            elif k == "d":
                command[1] = max(command[1] - 0.1, -0.35)
            elif k == "q":
                command[2] = min(command[2] + 0.1, 0.9)
            elif k == "e":
                command[2] = max(command[2] - 0.1, -0.9)
            elif k == " ":
                command[:] = 0.0
            if k:
                print(f"cmd vx {command[0]:+.1f} vy {command[1]:+.1f} "
                      f"wz {command[2]:+.1f}")

            joint_vel = (joint_pos - prev_joint) / dt
            obs = np.concatenate([
                gravity, zeros3, zeros3,
                joint_pos - home, joint_vel, prev_action, command,
            ]).astype(np.float32)

            action = agent.act_deterministic(obs)
            action = np.clip(action, -1.0, 1.0)
            target = np.clip(home + scale * action, jnt_lo, jnt_hi)

            prev_joint = joint_pos
            joint_pos = target          # open loop: assume servos track
            prev_action = action

            frame = servo_map.frame_line(
                servo_map.qpos_to_servo_deg(target, home))
            if ser:
                ser.write(frame)
            else:
                print(frame.decode().strip())

            leftover = dt - (time.time() - t0)
            if leftover > 0:
                time.sleep(leftover)
    finally:
        if ser:
            ser.write(b"H\n")
            ser.close()
            print("eased to home, port closed")


if __name__ == "__main__":
    main()
