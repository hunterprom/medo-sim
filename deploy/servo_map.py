"""Sim joint <-> physical servo mapping for Medo.

The channel/direction/trim data comes from the hand-written trot firmware
(deploy/reference/robotic_dog_walking_cycle.ino). Naming clash to keep in
mind: that sketch calls the leg pitch servo "hip" and holds the abduction
servos ("HIP_xx" constants) at fixed angles; the sim calls abduction
"<leg>_hip" and the pitch joint "<leg>_thigh".

Sim actuator order (matches assets/medo_real.xml):
    fr_hip fr_thigh fr_knee  fl_*  br_*  bl_*
Firmware leg names:  FR=fr  FL=fl  RR=br  RL=bl   (VERIFY on the dog: the
Fusion export's left/right labels may be mirrored; see deploy/README.md)

Every DIR sign and HOME_SERVO_DEG below is an initial guess taken from the
trot sketch's trims/dirs and its standing IK pose. They MUST be verified
joint by joint with the calibration procedure in deploy/README.md before
letting a policy drive the dog.
"""

from __future__ import annotations

import math

import numpy as np

# (board, channel): board 0 = PCA at 0x40 (front), 1 = PCA at 0x41 (rear)
CHANNELS = [
    (0, 11),  # fr_hip   (abduction; sketch "HIP_FR")
    (0, 4),   # fr_thigh (sketch FR hip)
    (0, 7),   # fr_knee
    (0, 10),  # fl_hip   (sketch "HIP_FL")
    (0, 1),   # fl_thigh
    (0, 2),   # fl_knee
    (1, 14),  # br_hip   (sketch "HIP_RR")
    (1, 0),   # br_thigh (sketch RR hip)
    (1, 2),   # br_knee
    (1, 15),  # bl_hip   (sketch "HIP_RL")
    (1, 5),   # bl_thigh (sketch RL hip)
    (1, 6),   # bl_knee
]

# servo degrees when the sim is at its "home" keyframe -- CALIBRATE THESE.
# Abduction homes = the sketch's fixed HIP_xx angles. Thigh/knee homes = the
# sketch's standing pose from its IK (trim 90, hip -13.3 deg, knee 31.1 deg).
HOME_SERVO_DEG = np.array([
    45.0, 76.7, 121.1,   # fr
    35.0, 103.3, 58.9,   # fl
    75.0, 76.7, 121.1,   # br
    45.0, 103.3, 58.9,   # bl
])

# +1 / -1 per joint: which way the servo turns for a positive sim joint
# angle -- CALIBRATE THESE (guesses follow the sketch's hip_dir/knee_dir).
DIR = np.array([
    +1.0, +1.0, -1.0,    # fr
    -1.0, -1.0, +1.0,    # fl
    +1.0, +1.0, -1.0,    # br
    -1.0, -1.0, +1.0,    # bl
])

SERVO_MIN_DEG = 5.0
SERVO_MAX_DEG = 175.0


def qpos_to_servo_deg(qpos_joints: np.ndarray, home_qpos: np.ndarray) -> np.ndarray:
    """12 sim joint angles (rad, actuator order) -> 12 servo degrees."""
    delta = np.degrees(np.asarray(qpos_joints) - np.asarray(home_qpos))
    deg = HOME_SERVO_DEG + DIR * delta
    return np.clip(deg, SERVO_MIN_DEG, SERVO_MAX_DEG)


def frame_line(servo_deg: np.ndarray) -> bytes:
    """Encode one bridge command frame: 'T d0 ... d11\\n' (0.1 deg precision)."""
    return ("T " + " ".join(f"{d:.1f}" for d in servo_deg) + "\n").encode()


def home_from_model(model) -> np.ndarray:
    """Joint home pose (rad, actuator order) from the model's home keyframe."""
    import mujoco
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key < 0:
        raise ValueError("model has no 'home' keyframe")
    return model.key_qpos[key][7:].copy()


if __name__ == "__main__":
    # print the home frame as a sanity check
    print(frame_line(HOME_SERVO_DEG).decode().strip())
    d = qpos_to_servo_deg(np.zeros(12) + math.radians(5), np.zeros(12))
    print(frame_line(d).decode().strip())
