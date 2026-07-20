"""Convert a URDF exported from Fusion 360 (ACDC4Robot) into a training-ready MJCF.

    python tools/convert_model.py path/to/medo.urdf --out assets/medo_real.xml

What it does:
  1. rewrites package:// mesh paths so MuJoCo can find the STLs
  2. compiles the URDF with MuJoCo and saves it as MJCF
  3. injects: floor + light + skybox, position actuators for every hinge
     joint, sim options (timestep/integrator), and a 'home' keyframe stub

What YOU still do afterwards (see docs/EXPORT_GUIDE.md):
  - rename the foot/toe geoms so their names contain "foot"
  - edit the 'home' keyframe to a good crouched stance
  - set kp / forcerange on the actuators to match Medo's real servos
  - sanity-check masses in `python view.py --model assets/medo_real.xml`
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import xml.etree.ElementTree as ET

import mujoco

ASSET_XML = """
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.1"/>
"""


def patch_urdf(urdf_path: str) -> str:
    """Strip package:// prefixes so mesh paths resolve relative to the URDF."""
    with open(urdf_path, encoding="utf-8") as f:
        text = f.read()
    patched = re.sub(r'filename="package://[^/]+/', 'filename="', text)
    patched = patched.replace('filename="../', 'filename="')
    fd, tmp = tempfile.mkstemp(suffix=".urdf", dir=os.path.dirname(urdf_path) or ".")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(patched)
    return tmp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("urdf")
    p.add_argument("--out", default=os.path.join("assets", "medo_real.xml"))
    p.add_argument("--kp", type=float, default=25.0, help="servo position gain")
    p.add_argument("--torque", type=float, default=4.5, help="servo torque limit (Nm)")
    args = p.parse_args()

    tmp_urdf = patch_urdf(args.urdf)
    try:
        model = mujoco.MjModel.from_xml_path(tmp_urdf)
        fd, tmp_xml = tempfile.mkstemp(suffix=".xml")
        os.close(fd)
        mujoco.mj_saveLastXML(tmp_xml, model)
    finally:
        os.remove(tmp_urdf)

    tree = ET.parse(tmp_xml)
    os.remove(tmp_xml)
    root = tree.getroot()

    # sim options
    for opt in root.findall("option"):
        root.remove(opt)
    ET.SubElement(root, "option", {"timestep": "0.005", "integrator": "implicitfast"})

    # assets + floor + light
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    asset.extend(ET.fromstring(f"<a>{ASSET_XML}</a>"))
    worldbody = root.find("worldbody")
    if worldbody.find("geom[@name='floor']") is None:
        ET.SubElement(worldbody, "geom", {
            "name": "floor", "type": "plane", "size": "0 0 0.05",
            "material": "groundplane", "friction": "1.0 0.005 0.0001"})
    ET.SubElement(worldbody, "light", {
        "pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"})

    # free joint for the base if the URDF was fixed to the world
    bodies = worldbody.findall("body")
    if bodies and bodies[0].find("freejoint") is None and bodies[0].find("joint") is None:
        bodies[0].insert(0, ET.Element("freejoint", {"name": "root"}))

    # position actuators for every hinge joint
    hinge_names = []
    for m in range(model.njnt):
        if model.jnt_type[m] == mujoco.mjtJoint.mjJNT_HINGE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, m)
            if name:
                hinge_names.append(name)
    for act in root.findall("actuator"):
        root.remove(act)
    actuator = ET.SubElement(root, "actuator")
    for name in hinge_names:
        ET.SubElement(actuator, "position", {
            "name": name, "joint": name, "kp": str(args.kp), "kv": "0.6",
            "inheritrange": "1",
            "forcerange": f"-{args.torque} {args.torque}"})

    # keyframe stub: all zeros, correct sizes -- edit to a crouched stance
    for key in root.findall("keyframe"):
        root.remove(key)
    keyframe = ET.SubElement(root, "keyframe")
    nq = model.nq + (7 if bodies and bodies[0].find("freejoint") is not None
                     and model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE else 0)
    ET.SubElement(keyframe, "key", {
        "name": "home",
        "qpos": " ".join(["0"] * nq),
        "ctrl": " ".join(["0"] * len(hinge_names))})

    ET.indent(tree)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tree.write(args.out, encoding="unicode")

    print(f"wrote {args.out}")
    print(f"  hinge joints found: {len(hinge_names)}")
    for n in hinge_names:
        print(f"    {n}")
    print("next steps:")
    print("  1. verify it loads:  python view.py --model " + args.out)
    print("  2. rename foot geoms to contain 'foot' (edit the XML)")
    print("  3. edit the 'home' keyframe: base z at standing height, quat 1 0 0 0,")
    print("     joints in a stable crouch; copy joint values into ctrl too")
    print("  4. set --kp/--torque (or edit XML) to match the real servos")
    print("  5. train:  python train.py --model " + args.out)


if __name__ == "__main__":
    main()
