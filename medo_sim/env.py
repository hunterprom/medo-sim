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
    # action
    action_scale: float = 0.5
    # observation layout: drop base linear velocity for policies meant to run
    # on the real dog (an IMU gives gravity + ang vel, but not lin vel)
    obs_lin_vel: bool = True
    # reward scales
    rew_track_lin: float = 1.0
    rew_track_ang: float = 0.5
    tracking_sigma: float = 0.15   # tight: standing must not keep half credit
    rew_lin_vel_z: float = -2.0
    rew_ang_vel_xy: float = -0.05
    rew_orientation: float = -1.0
    rew_torque: float = -2e-4
    rew_action_rate: float = -0.01
    rew_air_time: float = 0.5
    air_time_target_s: float = 0.3
    # anti-statue: with a movement command active, every foot beyond two on
    # the ground costs reward. air_time alone cannot break a standing policy
    # because it only pays on touchdowns, which a statue never produces.
    rew_feet_planted: float = -0.25
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


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v from world frame into the frame of quaternion q (wxyz)."""
    w, vec = q[0], q[1:]
    a = v * (2.0 * w * w - 1.0)
    b = 2.0 * w * np.cross(vec, v)
    c = 2.0 * vec * np.dot(vec, v)
    return a - b + c


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
                      + self.nu + self.nu + self.nu + 3)

        # default pose from the "home" keyframe (required)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id < 0:
            raise ValueError(f"model {self.cfg.model_path} needs a keyframe named 'home'")
        self.home_qpos = self.model.key_qpos[key_id].copy()
        self.default_joint_pos = self.home_qpos[7:].copy()

        self.torso_body = self._find_torso_body()
        self.floor_geom = self._find_floor_geom()
        # feet = robot geoms whose name contains "foot"; everything else on the
        # robot touching the floor terminates the episode
        foot, illegal = [], []
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] == 0:
                continue  # world geom
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            (foot if "foot" in name.lower() else illegal).append(gid)
        if not foot:
            raise ValueError("no geoms with 'foot' in their name; rename foot geoms")
        self.foot_geoms = np.array(foot)
        self.illegal_geoms = np.array(illegal)

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
        ])
        self._gravity_world = np.array([0.0, 0.0, -1.0])
        self._vel6 = np.zeros(6)

        self.command = np.zeros(3)
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
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.torso_body,
            self._vel6, 1,
        )
        self.ang_vel = self._vel6[:3].copy()
        self.lin_vel = self._vel6[3:].copy()
        self.gravity = quat_rotate_inverse(self.data.qpos[3:7], self._gravity_world)
        # contacts against the floor, vectorized over the contact buffer
        ncon = self.data.ncon
        self.foot_contact = np.zeros(len(self.foot_geoms), dtype=bool)
        self.illegal_contact = False
        if ncon:
            g1 = self.data.contact.geom1[:ncon]
            g2 = self.data.contact.geom2[:ncon]
            on_floor = (g1 == self.floor_geom) | (g2 == self.floor_geom)
            if on_floor.any():
                other = np.where(g1[on_floor] == self.floor_geom,
                                 g2[on_floor], g1[on_floor])
                self.illegal_contact = bool(np.isin(other, self.illegal_geoms).any())
                self.foot_contact = np.isin(self.foot_geoms, other)

    def _sample_command(self) -> np.ndarray:
        c = self.cfg
        if self.rng.random() < c.zero_cmd_prob:
            return np.zeros(3)
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
        self.air_time += self.dt
        self.air_time[contact] = 0.0
        self.prev_contact = contact

        return float(sum(comps.values())), {k: float(v) for k, v in comps.items()}
