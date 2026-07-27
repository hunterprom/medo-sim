"""Quadruped locomotion environment on MuJoCo.

One MedoEnv owns one MjModel/MjData pair, so many envs can step in parallel
threads (MuJoCo releases the GIL inside mj_step). The task is velocity-command
following: the policy gets a body-frame command (vx, vy, wz) and is rewarded
for tracking it, legged-gym style.

Observation (48,):
    projected gravity (3), base ang vel (3), base lin vel (3),
    joint pos - default (12), joint vel (12), previous action (12), command (3)
Action (12,): offsets from the default joint pose, scaled by cfg.action_scale,
    fed to MuJoCo position actuators (PD servos).

The per-step Python work runs under the GIL and is what actually limits
throughput with many threads, so it is kept deliberately lean: contacts are
checked with vectorized numpy, base velocities / gravity are computed once per
control step, and observation noise is a single pre-scaled draw.
"""

from __future__ import annotations

import dataclasses
import os

import mujoco
import numpy as np

_ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)
# the real Medo (built from the Fusion 360 export) when present, else the
# placeholder quadruped
DEFAULT_MODEL = os.path.join(_ASSETS, "medo_real.xml")
if not os.path.exists(DEFAULT_MODEL):
    DEFAULT_MODEL = os.path.join(_ASSETS, "medo.xml")


@dataclasses.dataclass
class EnvConfig:
    model_path: str = DEFAULT_MODEL
    # timing
    decimation: int = 4              # sim steps per control step (200 Hz -> 50 Hz)
    episode_length_s: float = 20.0
    # commands (body frame), sized for the ~1.5 kg MG996 dog: symmetric-ish
    # coverage of forward/backward/strafe/turn at speeds it can actually hit
    cmd_vx_range: tuple = (-0.5, 0.7)
    cmd_vy_range: tuple = (-0.35, 0.35)
    cmd_wz_range: tuple = (-0.9, 0.9)
    cmd_resample_s: float = 5.0
    zero_cmd_prob: float = 0.1       # some episodes learn to stand still
    cmd_straight: bool = False       # straight-line specialist: vy = wz = 0
    # action
    action_scale: float = 0.5
    # observation layout: drop base linear velocity for policies meant to run
    # on the real dog (an IMU gives gravity + ang vel, but not lin vel)
    obs_lin_vel: bool = True
    # gait clock: append (sin, cos) of a free-running phase. Without it the
    # policy has no time reference -- near the home pose identical
    # observations demand different actions depending on gait phase, which
    # defeats both imitation and deterministic gait startup. The real robot
    # synthesizes the same clock from a timer.
    obs_gait_clock: bool = False
    gait_clock_hz: float = 1.0
    # heading error observation: sin/cos of (yaw - commanded heading). The
    # deploy obs has no compass, so without this the policy can only keep its
    # AVERAGE yaw rate at zero -- any residual asymmetry integrates into a
    # slow circle it cannot see. On the real dog this signal is the MPU6050
    # gyro wz integral minus the commanded wz integral: deployable, unlike
    # absolute heading.
    obs_heading_err: bool = False
    # reward scales
    rew_track_lin: float = 1.0
    rew_track_ang: float = 0.5
    tracking_sigma: float = 0.15   # tight: standing must not keep half credit
    # linear directional progress: pays from the first cm/s toward the
    # command (capped at the command). The exp() tracking well is too flat
    # far from target to out-bid fall risk with few exploration envs --
    # observed: a marching-in-place equilibrium at zero velocity.
    rew_progress: float = 0.7
    rew_ang_progress: float = 0.35
    rew_lin_vel_z: float = -2.0
    rew_ang_vel_xy: float = -0.05
    rew_orientation: float = -1.0
    rew_torque: float = -2e-4
    rew_action_rate: float = -0.01
    rew_air_time: float = 0.3   # kept modest: air time alone rewards marching
    air_time_target_s: float = 0.3
    # anti-statue: with a movement command active, every foot beyond two on
    # the ground costs reward. air_time alone cannot break a standing policy
    # because it only pays on touchdowns, which a statue never produces.
    rew_feet_planted: float = -0.25
    # lower-limb floor grazes are penalized, not terminal: a working trot on
    # this 11 cm-leg dog brushes a thigh capsule ~1% of steps, and killing
    # the episode for that teaches the policy that walking is lethal
    rew_limb_contact: float = -0.15
    # body/torso floor contact still terminates, but ALSO costs points on
    # that final step: only the toes may ever touch the ground for free
    rew_body_contact: float = -1.0
    # --- straight-walk shaping (all off by default; train.py --straight) ---
    # heading hold: squared error between actual yaw and the commanded
    # heading (initial yaw + integral of commanded wz). The policy cannot
    # observe absolute heading, but zero net yaw per gait cycle is learnable
    # and is exactly what stops veering-into-circles.
    rew_heading: float = 0.0
    # velocity across the commanded track, in the WORLD frame: a robot that
    # has yawed but keeps body vy = 0 walks a circle without body-frame
    # tracking ever noticing
    rew_lateral: float = 0.0
    # trot symmetry: diagonal foot pairs (fr+bl, fl+br) must displace alike
    # (y mirrored). Measured at the feet, because the export's per-leg joint
    # sign conventions are chirally inconsistent and joint-space symmetry
    # would lie.
    rew_symmetry: float = 0.0
    # stance anchoring: quadratic pull toward the home keyframe; hips (not
    # needed for a straight trot) pinned much harder
    rew_pose: float = 0.0
    rew_hip_pose: float = 0.0
    # domain randomization
    randomize: bool = True
    friction_range: tuple = (0.5, 1.25)
    mass_scale_range: tuple = (0.85, 1.2)
    push_interval_s: float = 4.0
    push_vel: float = 0.5
    # observation noise
    obs_noise: bool = True
    noise_gravity: float = 0.05
    noise_ang_vel: float = 0.1
    noise_lin_vel: float = 0.1
    noise_joint_pos: float = 0.01
    noise_joint_vel: float = 0.15
    # termination
    max_tilt_gz: float = -0.5        # terminate if projected gravity z > this (>60 deg tilt)


def layout_for_n_obs(n_obs: int, nu: int = 12) -> dict:
    """Map a checkpoint's obs size to EnvConfig layout flags."""
    base = 3 + 3 + 3 * nu + 3
    layouts = {
        base + 3: dict(obs_lin_vel=True, obs_gait_clock=False),
        base: dict(obs_lin_vel=False, obs_gait_clock=False),
        base + 5: dict(obs_lin_vel=True, obs_gait_clock=True),
        base + 2: dict(obs_lin_vel=False, obs_gait_clock=True),
        base + 4: dict(obs_lin_vel=False, obs_gait_clock=True,
                       obs_heading_err=True),
        base + 7: dict(obs_lin_vel=True, obs_gait_clock=True,
                       obs_heading_err=True),
    }
    if n_obs not in layouts:
        raise ValueError(f"no known obs layout has size {n_obs} for {nu} joints")
    return layouts[n_obs]


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v from world frame into the frame of quaternion q (wxyz)."""
    w, vec = q[0], q[1:]
    a = v * (2.0 * w * w - 1.0)
    b = 2.0 * w * np.cross(vec, v)
    c = 2.0 * vec * np.dot(vec, v)
    return a - b + c


def quat_rotate_inverse_mat(q: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Rotate rows of V (N, 3) from world frame into the frame of q (wxyz)."""
    w, vec = q[0], q[1:]
    a = V * (2.0 * w * w - 1.0)
    b = 2.0 * w * np.cross(vec, V)
    c = 2.0 * vec * (V @ vec)[:, None]
    return a - b + c


def yaw_of(q: np.ndarray) -> float:
    """Yaw (world Z rotation) of quaternion q (wxyz)."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class MedoEnv:
    def __init__(self, cfg: EnvConfig | None = None, seed: int = 0):
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_path(self.cfg.model_path)
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep * self.cfg.decimation
        self.max_steps = int(self.cfg.episode_length_s / self.dt)
        self.cmd_resample_steps = int(self.cfg.cmd_resample_s / self.dt)
        self.push_interval_steps = max(1, int(self.cfg.push_interval_s / self.dt))

        self.nu = self.model.nu
        self.n_obs = (3 + 3 + (3 if self.cfg.obs_lin_vel else 0)
                      + self.nu + self.nu + self.nu + 3
                      + (2 if self.cfg.obs_gait_clock else 0)
                      + (2 if self.cfg.obs_heading_err else 0))

        # default pose from the "home" keyframe (required)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id < 0:
            raise ValueError(f"model {self.cfg.model_path} needs a keyframe named 'home'")
        self.home_qpos = self.model.key_qpos[key_id].copy()
        self.default_joint_pos = self.home_qpos[7:].copy()

        self.torso_body = self._find_torso_body()
        self.floor_geom = self._find_floor_geom()
        # feet ("foot" in name) may touch the floor; thigh/shank contacts are
        # penalized but survivable; anything else touching terminates
        foot, limb, terminal = [], [], []
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] == 0:
                continue  # world geom
            name = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                    or "").lower()
            if "foot" in name:
                foot.append(gid)
            elif "thigh" in name or "shank" in name:
                limb.append(gid)
            else:
                terminal.append(gid)
        if not foot:
            raise ValueError("no geoms with 'foot' in their name; rename foot geoms")
        self.foot_geoms = np.array(foot)
        self.limb_geoms = np.array(limb)
        self.illegal_geoms = np.array(terminal)

        # symmetry term: foot order fr, fl, br, bl by geom name; disabled if
        # the model lacks these names (e.g. the placeholder quadruped)
        order = []
        for leg in ("fr", "fl", "br", "bl"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"{leg}_foot")
            if gid >= 0 and gid in foot:
                order.append(int(np.where(self.foot_geoms == gid)[0][0]))
        self._sym_feet = np.array(order) if len(order) == 4 else None
        if self._sym_feet is not None:
            d = mujoco.MjData(self.model)
            d.qpos[:] = self.model.key_qpos[key_id]
            mujoco.mj_kinematics(self.model, d)
            self._home_foot_off = quat_rotate_inverse_mat(
                self.home_qpos[3:7],
                d.geom_xpos[self.foot_geoms[self._sym_feet]] - self.home_qpos[0:3])
        self._hip_idx = np.arange(0, self.nu, 3)  # joint order: <leg>_hip first

        # joint ranges for clipping position targets
        self.jnt_lo = self.model.jnt_range[1:, 0].copy()
        self.jnt_hi = self.model.jnt_range[1:, 1].copy()

        # base values for domain randomization
        self.base_body_mass = self.model.body_mass.copy()
        self.base_body_inertia = self.model.body_inertia.copy()

        # one pre-scaled noise vector; a single rng draw per step covers all
        # noisy segments (prev_action and command stay exact)
        c = self.cfg
        self.noise_vec = np.concatenate([
            np.full(3, c.noise_gravity),
            np.full(3, c.noise_ang_vel),
            np.full(3, c.noise_lin_vel) if c.obs_lin_vel else np.zeros(0),
            np.full(self.nu, c.noise_joint_pos),
            np.full(self.nu, c.noise_joint_vel),
            np.zeros(self.nu),
            np.zeros(3),
            np.zeros(2 if c.obs_gait_clock else 0),
            # heading error: gyro-integration noise on the real dog
            np.full(2 if c.obs_heading_err else 0, c.noise_gravity),
        ])
        self._gravity_world = np.array([0.0, 0.0, -1.0])
        self._vel6 = np.zeros(6)

        self.command = np.zeros(3)
        self.gait_phase = 0.0
        self.heading_ref = 0.0
        self.prev_action = np.zeros(self.nu)
        self.air_time = np.zeros(len(self.foot_geoms))
        self.prev_contact = np.zeros(len(self.foot_geoms), dtype=bool)
        self.step_count = 0
        self.ep_reward_sums: dict[str, float] = {}
        self.ep_return = 0.0

    # ------------------------------------------------------------- helpers

    def _find_torso_body(self) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        if bid >= 0:
            return bid
        # fall back to the body the free joint hangs off
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return self.model.jnt_bodyid[j]
        raise ValueError("no body named 'torso' and no free joint found")

    def _find_floor_geom(self) -> int:
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if gid >= 0:
            return gid
        for g in range(self.model.ngeom):
            if (self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
                    and self.model.geom_bodyid[g] == 0):
                return g
        raise ValueError("no floor: add a plane geom named 'floor'")

    def _refresh_state(self):
        """Compute the per-step quantities shared by reward/termination/obs."""
        # Base velocities straight from the free joint: qvel[0:3] is the base
        # linear velocity in WORLD coords, qvel[3:6] the angular velocity
        # already in BODY coords. (mj_objectVelocity was measurably NOT
        # returning body-frame base velocity here -- a walking robot read as
        # standing still, which silently zeroed every velocity reward.)
        self.lin_vel = quat_rotate_inverse(self.data.qpos[3:7],
                                           self.data.qvel[0:3])
        self.ang_vel = self.data.qvel[3:6].copy()
        self.gravity = quat_rotate_inverse(self.data.qpos[3:7], self._gravity_world)
        self.yaw = yaw_of(self.data.qpos[3:7])
        # contacts against the floor, vectorized over the contact buffer
        ncon = self.data.ncon
        self.foot_contact = np.zeros(len(self.foot_geoms), dtype=bool)
        self.illegal_contact = False
        self.limb_contacts = 0
        if ncon:
            g1 = self.data.contact.geom1[:ncon]
            g2 = self.data.contact.geom2[:ncon]
            on_floor = (g1 == self.floor_geom) | (g2 == self.floor_geom)
            if on_floor.any():
                other = np.where(g1[on_floor] == self.floor_geom,
                                 g2[on_floor], g1[on_floor])
                self.illegal_contact = bool(np.isin(other, self.illegal_geoms).any())
                self.limb_contacts = int(np.isin(self.limb_geoms, other).sum())
                self.foot_contact = np.isin(self.foot_geoms, other)

    def _sample_command(self) -> np.ndarray:
        c = self.cfg
        if self.rng.random() < c.zero_cmd_prob:
            return np.zeros(3)
        if c.cmd_straight:
            return np.array([self.rng.uniform(*c.cmd_vx_range), 0.0, 0.0])
        return np.array([
            self.rng.uniform(*c.cmd_vx_range),
            self.rng.uniform(*c.cmd_vy_range),
            self.rng.uniform(*c.cmd_wz_range),
        ])

    # ------------------------------------------------------------- API

    def reset(self) -> np.ndarray:
        c = self.cfg
        if c.randomize:
            self.model.geom_friction[self.floor_geom, 0] = self.rng.uniform(*c.friction_range)
            scale = self.rng.uniform(*c.mass_scale_range)
            self.model.body_mass[:] = self.base_body_mass
            self.model.body_inertia[:] = self.base_body_inertia
            self.model.body_mass[self.torso_body] *= scale
            self.model.body_inertia[self.torso_body] *= scale

        mujoco.mj_resetData(self.model, self.data)
        qpos = self.home_qpos.copy()
        qpos[7:] += self.rng.uniform(-0.05, 0.05, size=self.nu)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = self.rng.uniform(-0.05, 0.05, size=self.model.nv)
        self.data.ctrl[:] = qpos[7:]
        mujoco.mj_forward(self.model, self.data)

        self.command = self._sample_command()
        self.gait_phase = float(self.rng.random())  # random phase: no fixed t=0
        self.heading_ref = yaw_of(self.data.qpos[3:7])
        self.prev_action[:] = 0.0
        self.air_time[:] = 0.0
        self.prev_contact[:] = False
        self.step_count = 0
        self.ep_reward_sums = {}
        self.ep_return = 0.0
        self._refresh_state()
        return self._observe()

    def step(self, action: np.ndarray):
        c = self.cfg
        action = np.clip(action, -1.0, 1.0)
        target = np.clip(
            self.default_joint_pos + c.action_scale * action, self.jnt_lo, self.jnt_hi
        )
        self.data.ctrl[:] = target
        for _ in range(c.decimation):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.gait_phase = (self.gait_phase + self.dt * c.gait_clock_hz) % 1.0
        # commanded heading advances with the wz command active this step
        self.heading_ref += self.command[2] * self.dt

        if c.randomize and self.step_count % self.push_interval_steps == 0:
            self.data.qvel[0:2] += self.rng.uniform(-c.push_vel, c.push_vel, size=2)

        if self.step_count % self.cmd_resample_steps == 0:
            self.command = self._sample_command()

        self._refresh_state()
        reward, comps = self._reward(action)
        self.ep_return += reward
        for k, v in comps.items():
            self.ep_reward_sums[k] = self.ep_reward_sums.get(k, 0.0) + v

        terminated = bool(self.gravity[2] > c.max_tilt_gz) or self.illegal_contact
        truncated = self.step_count >= self.max_steps and not terminated

        obs = self._observe()
        self.prev_action = action.copy()
        info = {}
        if terminated or truncated:
            n = max(1, self.step_count)
            info["episode"] = {
                "return": self.ep_return,
                "length": self.step_count,
                **{f"rew_{k}": v / n for k, v in self.ep_reward_sums.items()},
            }
        return obs, reward, terminated, truncated, info

    def _observe(self) -> np.ndarray:
        segs = [self.gravity, self.ang_vel]
        if self.cfg.obs_lin_vel:
            segs.append(self.lin_vel)
        segs += [
            self.data.qpos[7:] - self.default_joint_pos,
            self.data.qvel[6:],
            self.prev_action, self.command,
        ]
        if self.cfg.obs_gait_clock:
            ph = 2.0 * np.pi * self.gait_phase
            segs.append(np.array([np.sin(ph), np.cos(ph)]))
        if self.cfg.obs_heading_err:
            herr = (self.yaw - self.heading_ref + np.pi) % (2.0 * np.pi) - np.pi
            segs.append(np.array([np.sin(herr), np.cos(herr)]))
        obs = np.concatenate(segs)
        if self.cfg.obs_noise:
            obs += self.rng.uniform(-1.0, 1.0, self.n_obs) * self.noise_vec
        return obs.astype(np.float32)

    def _reward(self, action: np.ndarray) -> tuple[float, dict]:
        c = self.cfg
        torques = self.data.qfrc_actuator[6:]
        lin_err = np.sum((self.command[:2] - self.lin_vel[:2]) ** 2)
        ang_err = (self.command[2] - self.ang_vel[2]) ** 2

        comps = {
            "track_lin": c.rew_track_lin * np.exp(-lin_err / c.tracking_sigma),
            "track_ang": c.rew_track_ang * np.exp(-ang_err / c.tracking_sigma),
            "lin_vel_z": c.rew_lin_vel_z * self.lin_vel[2] ** 2,
            "ang_vel_xy": c.rew_ang_vel_xy * np.sum(self.ang_vel[:2] ** 2),
            "orientation": c.rew_orientation * np.sum(self.gravity[:2] ** 2),
            "torque": c.rew_torque * np.sum(torques ** 2),
            "action_rate": c.rew_action_rate * np.sum((action - self.prev_action) ** 2),
        }

        # reward long swing phases: paid once per touchdown, only when moving
        contact = self.foot_contact
        touchdown = contact & ~self.prev_contact
        moving_cmd = np.linalg.norm(self.command) > 0.1
        if moving_cmd and touchdown.any():
            comps["air_time"] = c.rew_air_time * float(
                np.sum(self.air_time[touchdown] - c.air_time_target_s)
            )
        if moving_cmd:
            comps["feet_planted"] = c.rew_feet_planted * max(
                0.0, float(contact.sum()) - 2.0)
        if self.limb_contacts:
            comps["limb_contact"] = c.rew_limb_contact * self.limb_contacts
        if self.illegal_contact:
            comps["body_contact"] = c.rew_body_contact
        cmd_xy = self.command[:2]
        n_xy = float(np.linalg.norm(cmd_xy))
        if n_xy > 0.1:
            if c.cmd_straight:
                # pay only for advance along the commanded WORLD-frame track:
                # walking a curve with the nose forward earns less by
                # construction, so circling is never full credit
                t = np.array([np.cos(self.heading_ref),
                              np.sin(self.heading_ref)])
                proj = float(self.data.qvel[0:2] @ t) * np.sign(self.command[0])
            else:
                proj = float(self.lin_vel[:2] @ cmd_xy) / n_xy
            comps["progress"] = (c.rew_progress
                                 * float(np.clip(proj, -n_xy, n_xy)) / n_xy)
        n_wz = abs(self.command[2])
        if n_wz > 0.1:
            projw = float(self.ang_vel[2]) * np.sign(self.command[2])
            comps["ang_progress"] = (c.rew_ang_progress
                                     * float(np.clip(projw, -n_wz, n_wz)) / n_wz)

        if c.rew_pose or c.rew_hip_pose:
            dq = self.data.qpos[7:] - self.default_joint_pos
            if c.rew_pose:
                comps["pose"] = c.rew_pose * float(dq @ dq)
            if c.rew_hip_pose:
                hq = dq[self._hip_idx]
                comps["hip_pose"] = c.rew_hip_pose * float(hq @ hq)
        if c.rew_heading or c.rew_lateral:
            if c.rew_heading:
                err = (self.yaw - self.heading_ref + np.pi) \
                    % (2.0 * np.pi) - np.pi
                comps["heading"] = c.rew_heading * err * err
            if c.rew_lateral:
                v_lat = (-np.sin(self.heading_ref) * self.data.qvel[0]
                         + np.cos(self.heading_ref) * self.data.qvel[1])
                comps["lateral"] = c.rew_lateral * v_lat * v_lat
        if c.rew_symmetry and self._sym_feet is not None:
            off = quat_rotate_inverse_mat(
                self.data.qpos[3:7],
                self.data.geom_xpos[self.foot_geoms[self._sym_feet]]
                - self.data.qpos[0:3]) - self._home_foot_off
            # rows: fr, fl, br, bl. Trot diagonals (fr+bl, fl+br) displace
            # alike with y mirrored across the midline.
            m = np.array([1.0, -1.0, 1.0])
            e1 = off[0] - off[3] * m
            e2 = off[1] - off[2] * m
            comps["symmetry"] = c.rew_symmetry * float(e1 @ e1 + e2 @ e2)
        self.air_time += self.dt
        self.air_time[contact] = 0.0
        self.prev_contact = contact

        return float(sum(comps.values())), {k: float(v) for k, v in comps.items()}
