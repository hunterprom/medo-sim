"""Build assets/medo_real.xml from the Fusion 360 export of Medo.

The export (assets/robot_dog_description) has exact mesh placements and joint
axes that sit on the real MG996 servo shafts (verified to ~1 mm), but its
inertials are junk: every part was exported with Fusion's Steel material
(8.04 kg total, off-diagonal inertia terms that are not even positive
definite). So this builder:

  - keeps the URDF kinematics (joint origins/axes/limits) verbatim,
  - re-expresses everything in a torso frame with +X = forward (URDF front
    is -Y), origin at the hip centroid,
  - lets MuJoCo recompute mass/inertia from the meshes: printed parts at an
    effective PLA density, each MG996 pinned to its 55 g datasheet mass,
    plus a battery/electronics block in the torso,
  - adds hidden primitive collision geoms (box torso, capsules fitted to the
    leg meshes, toe spheres at the mesh foot tips) so contact stays fast,
  - models the servos honestly: 1.3 Nm force cap, velocity gain that tops
    out near the MG996's real loaded speed,
  - solves a standing "home" keyframe by FK grid search per leg.

    python tools/build_medo_real.py            # writes assets/medo_real.xml
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "assets", "robot_dog_description")
URDF = os.path.join(PKG, "urdf", "robot_dog.urdf")
OUT = os.path.join(ROOT, "assets", "medo_real.xml")
MESH_SCALE = 0.001                      # export is in mm

PLA_EFF = 750.0                         # kg/m^3: PLA shells + infill
SERVO_MASS = 0.055                      # kg, MG996R datasheet
BATTERY_MASS = 0.22                     # kg allowance: 2S lipo + driver + wiring
BATTERY_HALF = (0.045, 0.030, 0.015)    # m, box half-extents

SERVO_KP = 15.0                         # ~full torque at 5 deg error
SERVO_KV = 0.25                         # caps speed near 1.3/0.25 = 5.2 rad/s
SERVO_TORQUE = 1.3                      # N*m, MG996R stall ~1.1 at 6 V
JOINT_DAMPING = 0.05
JOINT_ARMATURE = 0.008
JOINT_FRICTION = 0.03

STAND_HEIGHT = 0.105                    # hip axis above foot center at home
FOOT_RADIUS = 0.008

LEGS = ["fr", "fl", "br", "bl"]         # declaration order (joints+actuators)

# torso frame: rotate old coords by +90 deg about z so old -Y (front) -> +X
R_FWD = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat_to_quat_str(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return " ".join(f"{x:.6f}" for x in q)


def vec_str(v, prec=6):
    return " ".join(f"{x:.{prec}f}" for x in v)


def load_stl_verts(name):
    with open(os.path.join(PKG, "meshes", name), "rb") as f:
        f.seek(80)
        n = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        rec = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    tri = rec[:, 12:48].copy().view(np.float32).reshape(n, 3, 3).astype(np.float64)
    return np.unique(tri.reshape(-1, 3), axis=0) * MESH_SCALE  # meters, local


# ---------------------------------------------------------------- parse URDF

def parse_urdf():
    robot = ET.parse(URDF).getroot()
    links, joints = {}, {}
    for link in robot.findall("link"):
        visuals = []
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh").get("filename").split("/")[-1]
            org = vis.find("origin")
            xyz = np.array([float(x) for x in org.get("xyz").split()])
            rpy = [float(x) for x in org.get("rpy").split()]
            visuals.append((mesh, xyz, rpy_to_R(*rpy)))
        links[link.get("name")] = visuals
    for j in robot.findall("joint"):
        lim = j.find("limit")
        joints[j.get("name")] = {
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": np.array([float(x) for x in j.find("origin").get("xyz").split()]),
            "axis": np.array([float(x) for x in j.find("axis").get("xyz").split()]),
            "range": (float(lim.get("lower")), float(lim.get("upper"))),
        }
    return links, joints


# ----------------------------------------------------- geometry helper fits

def link_verts(links, lname):
    """All mesh vertices of a link, transformed into the link frame."""
    chunks = []
    for mesh, xyz, R in links[lname]:
        v = load_stl_verts(mesh)
        chunks.append((R @ v.T).T + xyz)
    return np.vstack(chunks)


def fit_capsule(verts, r_lo=0.008, r_hi=0.018):
    """Capsule along the principal axis of a vertex cloud: (fromto, radius)."""
    c = verts.mean(0)
    _, _, vt = np.linalg.svd(verts - c, full_matrices=False)
    u = vt[0]
    proj = (verts - c) @ u
    lat = np.linalg.norm((verts - c) - np.outer(proj, u), axis=1)
    r = float(np.clip(np.percentile(lat, 80), r_lo, r_hi))
    half = max(0.5 * (proj.max() - proj.min()) - r, 0.005)
    a, b = c - u * half, c + u * half
    return np.concatenate([a, b]), r


def foot_tip(links, shank):
    """Toe sphere center in the shank frame: centroid of the far-tip cluster."""
    verts = link_verts(links, shank)
    d = np.linalg.norm(verts, axis=1)
    tip = verts[np.argmax(d)]
    cluster = verts[np.linalg.norm(verts - tip, axis=1) < 0.008]
    return cluster.mean(0)


# ---------------------------------------------------------------- build xml

def build(links, joints):
    hips = {leg: joints[f"{leg}_hip"]["xyz"] for leg in LEGS}
    center = np.mean(list(hips.values()), axis=0)

    def to_torso(p):
        return R_FWD @ (p - center)

    mj = ET.Element("mujoco", model="medo_real")
    ET.SubElement(mj, "compiler", angle="radian", autolimits="true",
                  meshdir="robot_dog_description/meshes")
    ET.SubElement(mj, "option", timestep="0.005", integrator="implicitfast")

    visual = ET.SubElement(mj, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6",
                  ambient="0.3 0.3 0.3", specular="0 0 0")
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(visual, "global", azimuth="130", elevation="-20",
                  offwidth="1280", offheight="720")
    ET.SubElement(visual, "quality", shadowsize="4096")

    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
    ET.SubElement(asset, "texture", type="2d", name="groundplane",
                  builtin="checker", mark="edge", rgb1="0.2 0.3 0.4",
                  rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8",
                  width="300", height="300")
    ET.SubElement(asset, "material", name="groundplane", texture="groundplane",
                  texuniform="true", texrepeat="5 5", reflectance="0.1")
    ET.SubElement(asset, "material", name="shell", rgba="0.75 0.75 0.78 1")
    ET.SubElement(asset, "material", name="dark", rgba="0.2 0.2 0.22 1")
    ET.SubElement(asset, "material", name="accent", rgba="0.9 0.5 0.1 1")
    meshes = sorted({m for vs in links.values() for m, _, _ in vs})
    for m in meshes:
        ET.SubElement(asset, "mesh", name=m[:-4], file=m,
                      scale="0.001 0.001 0.001", inertia="exact")

    default = ET.SubElement(mj, "default")
    d = ET.SubElement(default, "default", {"class": "medo"})
    ET.SubElement(d, "joint", damping=str(JOINT_DAMPING),
                  armature=str(JOINT_ARMATURE), frictionloss=str(JOINT_FRICTION))
    ET.SubElement(d, "position", kp=str(SERVO_KP), kv=str(SERVO_KV),
                  inheritrange="1",
                  forcerange=f"-{SERVO_TORQUE} {SERVO_TORQUE}")
    # visual-only meshes: no contact, no inertia contribution unless given mass
    dv = ET.SubElement(d, "default", {"class": "visual"})
    ET.SubElement(dv, "geom", type="mesh", contype="0", conaffinity="0",
                  group="1", material="shell", density=str(PLA_EFF))
    dvs = ET.SubElement(dv, "default", {"class": "servo"})
    ET.SubElement(dvs, "geom", material="dark", mass=str(SERVO_MASS))
    # collision primitives: hit the floor only (no self-collision), carry no mass
    dc = ET.SubElement(d, "default", {"class": "collision"})
    ET.SubElement(dc, "geom", contype="1", conaffinity="0", group="3",
                  mass="0", friction="1.0 0.005 0.0001", rgba="0.9 0.5 0.1 0.5")

    world = ET.SubElement(mj, "worldbody")
    ET.SubElement(world, "light", pos="0 0 1.5", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05",
                  material="groundplane", contype="1", conaffinity="1",
                  friction="1.0 0.005 0.0001")

    # torso: all base_link content re-expressed in the forward-x frame
    torso = ET.SubElement(world, "body", name="torso", pos="0 0 0.30",
                          childclass="medo")
    ET.SubElement(torso, "freejoint", name="root")
    ET.SubElement(torso, "site", name="imu", pos="0 0 0", size="0.01")
    base_pts = []
    for i, (mesh, xyz, R) in enumerate(links["base_link"]):
        cls = "servo" if "Tower_Pro" in mesh else "visual"
        ET.SubElement(torso, "geom", name=f"base_vis_{i}", mesh=mesh[:-4],
                      pos=vec_str(to_torso(xyz)),
                      quat=mat_to_quat_str(R_FWD @ R), **{"class": cls})
        v = load_stl_verts(mesh)
        base_pts.append((R_FWD @ ((R @ v.T).T + xyz - center).T).T)
    base_pts = np.vstack(base_pts)
    bb_min, bb_max = base_pts.min(0), base_pts.max(0)
    half = (bb_max - bb_min) / 2.0
    ET.SubElement(torso, "geom", name="torso_box", type="box",
                  pos=vec_str((bb_min + bb_max) / 2.0), size=vec_str(half),
                  **{"class": "collision"})
    ET.SubElement(torso, "geom", name="battery", type="box",
                  pos=vec_str((bb_min + bb_max) / 2.0),
                  size=vec_str(BATTERY_HALF), contype="0", conaffinity="0",
                  group="4", mass=str(BATTERY_MASS), rgba="0.1 0.1 0.1 0")

    for leg in LEGS:
        hip_j = joints[f"{leg}_hip"]
        thigh_j = joints[f"{leg}_thigh"]
        knee_j = joints[f"{leg}_knee"]
        shoulder, thigh, shank = hip_j["child"], thigh_j["child"], knee_j["child"]

        b_sh = ET.SubElement(torso, "body", name=shoulder,
                             pos=vec_str(to_torso(hip_j["xyz"])),
                             quat=mat_to_quat_str(R_FWD))
        ET.SubElement(b_sh, "joint", name=f"{leg}_hip",
                      axis=vec_str(hip_j["axis"], 0),
                      range=f"{hip_j['range'][0]} {hip_j['range'][1]}")
        add_link_geoms(b_sh, links, shoulder, leg)
        sh_verts = link_verts(links, shoulder)
        ET.SubElement(b_sh, "geom", name=f"{leg}_shoulder_col", type="sphere",
                      size="0.024", pos=vec_str(sh_verts.mean(0)),
                      **{"class": "collision"})

        b_th = ET.SubElement(b_sh, "body", name=thigh,
                             pos=vec_str(thigh_j["xyz"]))
        ET.SubElement(b_th, "joint", name=f"{leg}_thigh",
                      axis=vec_str(thigh_j["axis"], 0),
                      range=f"{thigh_j['range'][0]} {thigh_j['range'][1]}")
        add_link_geoms(b_th, links, thigh, leg)
        fromto, r = fit_capsule(link_verts(links, thigh))
        ET.SubElement(b_th, "geom", name=f"{leg}_thigh_col", type="capsule",
                      size=f"{r:.4f}", fromto=vec_str(fromto),
                      **{"class": "collision"})

        b_sk = ET.SubElement(b_th, "body", name=shank,
                             pos=vec_str(knee_j["xyz"]))
        ET.SubElement(b_sk, "joint", name=f"{leg}_knee",
                      axis=vec_str(knee_j["axis"], 0),
                      range=f"{knee_j['range'][0]} {knee_j['range'][1]}")
        add_link_geoms(b_sk, links, shank, leg)
        # fit the capsule to the shank minus the toe, so the named foot
        # sphere is the only geom that ever touches the floor
        sk_verts = link_verts(links, shank)
        tip0 = sk_verts[np.argmax(np.linalg.norm(sk_verts, axis=1))]
        sk_verts = sk_verts[np.linalg.norm(sk_verts - tip0, axis=1) > 0.018]
        fromto, r = fit_capsule(sk_verts, r_hi=0.012)
        ET.SubElement(b_sk, "geom", name=f"{leg}_shank_col", type="capsule",
                      size=f"{r:.4f}", fromto=vec_str(fromto),
                      **{"class": "collision"})
        tip = foot_tip(links, shank)
        ET.SubElement(b_sk, "geom", name=f"{leg}_foot", type="sphere",
                      size=str(FOOT_RADIUS), pos=vec_str(tip),
                      material="accent", **{"class": "collision"})

    act = ET.SubElement(mj, "actuator")
    for leg in LEGS:
        for jn in ["hip", "thigh", "knee"]:
            ET.SubElement(act, "position", name=f"{leg}_{jn}",
                          joint=f"{leg}_{jn}", **{"class": "medo"})

    sensor = ET.SubElement(mj, "sensor")
    ET.SubElement(sensor, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sensor, "accelerometer", name="imu_acc", site="imu")
    return mj


def add_link_geoms(body, links, lname, leg):
    for i, (mesh, xyz, R) in enumerate(links[lname]):
        cls = "servo" if "Tower_Pro" in mesh else "visual"
        ET.SubElement(body, "geom", name=f"{lname}_vis_{i}", mesh=mesh[:-4],
                      pos=vec_str(xyz), quat=mat_to_quat_str(R),
                      **{"class": cls})


# ------------------------------------------------------------- home keyframe

def solve_home(model):
    """Grid-search (thigh, knee) per leg: foot under hip at STAND_HEIGHT."""
    data = mujoco.MjData(model)
    torso_z = 0.5
    angles = {}
    for leg in LEGS:
        th_adr = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_thigh")]
        kn_adr = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_knee")]
        foot_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot")
        sh_bid = model.jnt_bodyid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_hip")]

        def cost_of(th, kn):
            mujoco.mj_resetData(model, data)
            data.qpos[2] = torso_z
            data.qpos[3] = 1.0
            data.qpos[th_adr] = th
            data.qpos[kn_adr] = kn
            mujoco.mj_kinematics(model, data)
            hip = data.xpos[sh_bid]
            foot = data.geom_xpos[foot_gid]
            dx = foot[0] - hip[0]
            dy = foot[1] - hip[1]
            dz = foot[2] - (hip[2] - STAND_HEIGHT)
            return dx * dx + 0.15 * dy * dy + dz * dz + 1e-4 * (th * th + kn * kn)

        lo, hi = -1.55, 1.55
        grid = np.linspace(lo, hi, 63)
        best = min(((cost_of(a, b), a, b) for a in grid for b in grid))
        _, ta, tb = best
        # local refine
        step = grid[1] - grid[0]
        for _ in range(3):
            fine_a = np.clip(np.linspace(ta - step, ta + step, 9), lo, hi)
            fine_b = np.clip(np.linspace(tb - step, tb + step, 9), lo, hi)
            _, ta, tb = min(((cost_of(a, b), a, b) for a in fine_a for b in fine_b))
            step /= 4.0
        angles[leg] = (ta, tb)

    # assemble qpos with all legs posed, drop torso so lowest foot touches floor
    mujoco.mj_resetData(model, data)
    data.qpos[2] = torso_z
    data.qpos[3] = 1.0
    for leg in LEGS:
        ta, tb = angles[leg]
        for jn, val in [("thigh", ta), ("knee", tb)]:
            adr = model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{jn}")]
            data.qpos[adr] = val
    mujoco.mj_kinematics(model, data)
    foot_low = min(
        data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                         f"{leg}_foot")][2]
        for leg in LEGS) - FOOT_RADIUS
    stand_z = torso_z - foot_low + 0.001
    qpos = data.qpos.copy()
    qpos[2] = stand_z
    return qpos, angles


def settle_check(model, qpos):
    """Step the model at the home pose; report drift."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.ctrl[:] = qpos[7:]
    mujoco.mj_forward(model, data)
    z0 = data.qpos[2]
    for _ in range(400):  # 2 s
        mujoco.mj_step(model, data)
    up = np.zeros(3)
    mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), data.qpos[3:7])
    return z0, data.qpos[2], up[2]


def compile_with_fallback(mj):
    """Compile OUT; meshes whose winding breaks inertia='exact' fall back to
    the convex-hull inertia (slightly overestimates that part's volume)."""
    import re
    for _ in range(len(list(mj.iter("mesh"))) + 1):
        ET.indent(mj)
        ET.ElementTree(mj).write(OUT)
        try:
            return mujoco.MjModel.from_xml_path(OUT)
        except ValueError as e:
            m = re.search(r"mesh '([^']+)'", str(e))
            if not m:
                raise
            bad = m.group(1)
            hits = [el for el in mj.iter("mesh") if el.get("name") == bad
                    and el.get("inertia") == "exact"]
            if not hits:
                raise
            hits[0].set("inertia", "convex")
            print(f"note: mesh {bad} not watertight, using convex-hull inertia")
    raise RuntimeError("mesh inertia fallback loop did not converge")


def main():
    links, joints = parse_urdf()
    mj = build(links, joints)
    model = compile_with_fallback(mj)
    qpos, angles = solve_home(model)

    key = ET.SubElement(mj, "keyframe")
    ET.SubElement(key, "key", name="home", qpos=vec_str(qpos, 4),
                  ctrl=vec_str(qpos[7:], 4))
    ET.indent(mj)
    ET.ElementTree(mj).write(OUT)

    model = mujoco.MjModel.from_xml_path(OUT)
    z0, z1, up_z = settle_check(model, qpos)

    print(f"wrote {OUT}")
    print(f"total mass: {model.body_mass.sum():.3f} kg")
    print(f"stand height (torso z): {qpos[2]:.4f} m")
    for leg in LEGS:
        ta, tb = angles[leg]
        print(f"  {leg}: thigh {ta:+.3f}  knee {tb:+.3f} rad")
    print(f"settle check (2 s servo hold): z {z0:.4f} -> {z1:.4f} m, "
          f"up-axis z {up_z:.3f}")
    if abs(z1 - z0) > 0.03 or up_z < 0.95:
        print("WARNING: home pose does not hold; inspect in view.py")


if __name__ == "__main__":
    main()
