"""Inspect a robot model in the interactive viewer (no policy needed).

    python view.py                     # placeholder dog
    python view.py --model assets/medo_real.xml

Prints a mass/joint/actuator summary, then opens the viewer posed at the
'home' keyframe if one exists. Use the Control tab sliders to move joints.
"""

from __future__ import annotations

import argparse

import mujoco
import mujoco.viewer

from medo_sim.env import DEFAULT_MODEL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    print(f"model: {args.model}")
    print(f"total mass: {sum(model.body_mass):.3f} kg")
    print(f"bodies: {model.nbody}  joints: {model.njnt}  geoms: {model.ngeom}  "
          f"actuators: {model.nu}")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            print(f"  joint {name}: free")
        else:
            lo, hi = model.jnt_range[j]
            print(f"  joint {name}: range [{lo:+.2f}, {hi:+.2f}] rad")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
