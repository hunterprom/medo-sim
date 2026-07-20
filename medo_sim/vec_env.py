"""Threaded vectorized wrapper around MedoEnv.

mj_step releases the GIL, so a plain thread pool scales across cores without
the pickling/spawn headaches of multiprocessing on Windows. Envs auto-reset;
when an episode is cut off by the time limit the pre-reset observation is
returned in final_obs so the learner can bootstrap correctly.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from medo_sim.env import EnvConfig, MedoEnv


class ThreadVecEnv:
    def __init__(self, num_envs: int, cfg: EnvConfig | None = None, seed: int = 0):
        self.num_envs = num_envs
        self.envs = [MedoEnv(cfg, seed=seed + i) for i in range(num_envs)]
        self.n_obs = self.envs[0].n_obs
        self.n_act = self.envs[0].nu
        self.pool = ThreadPoolExecutor(max_workers=num_envs)

    def reset(self) -> np.ndarray:
        obs = list(self.pool.map(lambda e: e.reset(), self.envs))
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        """Returns (obs, rewards, terminated, truncated, final_obs, ep_infos).

        final_obs[i] is the terminal observation for envs that ended this step
        (needed to bootstrap truncated episodes); obs[i] is already the reset
        observation for those envs.
        """
        results = list(self.pool.map(self._step_one, self.envs, actions))
        obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        terminated = np.array([r[2] for r in results], dtype=bool)
        truncated = np.array([r[3] for r in results], dtype=bool)
        final_obs = {i: r[4] for i, r in enumerate(results) if r[4] is not None}
        ep_infos = [r[5]["episode"] for r in results if "episode" in r[5]]
        return obs, rewards, terminated, truncated, final_obs, ep_infos

    @staticmethod
    def _step_one(env: MedoEnv, action: np.ndarray):
        obs, reward, terminated, truncated, info = env.step(action)
        final_obs = None
        if terminated or truncated:
            final_obs = obs
            obs = env.reset()
        return obs, reward, terminated, truncated, final_obs, info

    def close(self):
        self.pool.shutdown()
