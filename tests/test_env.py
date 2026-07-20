import numpy as np
import torch

from medo_sim.env import EnvConfig, MedoEnv
from medo_sim.ppo import PPO, PPOConfig
from medo_sim.vec_env import ThreadVecEnv


def test_env_reset_and_step():
    env = MedoEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (env.n_obs,)
    assert np.isfinite(obs).all()
    for _ in range(100):
        action = np.random.uniform(-1, 1, env.nu).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        if terminated or truncated:
            assert "episode" in info
            obs = env.reset()


def test_vec_env():
    venv = ThreadVecEnv(4, EnvConfig(), seed=0)
    obs = venv.reset()
    assert obs.shape == (4, venv.n_obs)
    for _ in range(20):
        actions = np.random.uniform(-1, 1, (4, venv.n_act)).astype(np.float32)
        obs, rewards, terminated, truncated, final_obs, ep_infos = venv.step(actions)
        assert obs.shape == (4, venv.n_obs)
        assert np.isfinite(obs).all()
    venv.close()


def test_ppo_update_runs():
    torch.manual_seed(0)
    venv = ThreadVecEnv(2, EnvConfig(), seed=0)
    agent = PPO(venv.n_obs, venv.n_act, PPOConfig(epochs=1, num_minibatches=2))
    T, N = 16, 2
    obs_raw = venv.reset()
    rows = {k: [] for k in ["obs", "actions", "logprobs", "mu", "sigma"]}
    rewards_l, dones_l, values_l = [], [], []
    for _ in range(T):
        obs_n, action, logp, value, mu, sigma = agent.act(obs_raw)
        obs_raw, rewards, term, trunc, _, _ = venv.step(action.numpy())
        rows["obs"].append(obs_n)
        rows["actions"].append(action)
        rows["logprobs"].append(logp)
        rows["mu"].append(mu)
        rows["sigma"].append(sigma)
        values_l.append(value)
        rewards_l.append(torch.from_numpy(rewards))
        dones_l.append(torch.from_numpy((term | trunc).astype(np.float32)))
    values = torch.stack(values_l)
    rewards = torch.stack(rewards_l)
    adv = rewards - values  # crude advantages are fine for a smoke test
    batch = {
        "obs": torch.cat(rows["obs"]),
        "actions": torch.cat(rows["actions"]),
        "logprobs": torch.cat(rows["logprobs"]),
        "mu": torch.cat(rows["mu"]),
        "sigma": torch.cat(rows["sigma"]),
        "advantages": adv.reshape(-1),
        "returns": (adv + values).reshape(-1),
    }
    stats = agent.update(batch)
    assert np.isfinite(stats["pi_loss"])
    assert np.isfinite(stats["kl"])
    venv.close()
