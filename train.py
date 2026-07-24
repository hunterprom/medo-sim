"""Train a walking policy for Medo with PPO over parallel MuJoCo envs.

    python train.py --run-name first_walk
    python train.py --resume runs/first_walk/ckpt_latest.pt

Logs to runs/<run-name>/log.csv, checkpoints to runs/<run-name>/ckpt_*.pt.
Ctrl+C saves a final checkpoint before exiting.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch

from medo_sim.env import DEFAULT_MODEL, EnvConfig
from medo_sim.ppo import PPO, PPOConfig
from medo_sim.vec_env import ThreadVecEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL, help="MJCF robot model path")
    p.add_argument("--run-name", default=time.strftime("run_%Y%m%d_%H%M%S"))
    p.add_argument("--num-envs", type=int, default=12)
    p.add_argument("--n-steps", type=int, default=256, help="rollout length per env")
    p.add_argument("--total-steps", type=int, default=30_000_000)
    p.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    p.add_argument("--deploy", action="store_true",
                   help="deployable observations only: drop base lin vel "
                        "(what the real dog's IMU cannot measure)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--ckpt-every", type=int, default=100, help="iterations between checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # tiny MLPs: many torch threads just fight the env threads for cores
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 3))

    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    env_cfg = EnvConfig(model_path=args.model, obs_lin_vel=not args.deploy)
    venv = ThreadVecEnv(args.num_envs, env_cfg, seed=args.seed)
    agent = PPO(venv.n_obs, venv.n_act, PPOConfig(), device=args.device)

    start_step = 0
    if args.resume:
        extra = agent.load(args.resume)
        start_step = extra.get("global_step", 0)
        print(f"resumed from {args.resume} at step {start_step}")

    # rollout buffers live on CPU; update() ships the batch to the GPU in one go
    N, T = args.num_envs, args.n_steps
    dev = torch.device("cpu")
    obs_buf = torch.zeros((T, N, venv.n_obs), device=dev)
    act_buf = torch.zeros((T, N, venv.n_act), device=dev)
    logp_buf = torch.zeros((T, N), device=dev)
    mu_buf = torch.zeros((T, N, venv.n_act), device=dev)
    sigma_buf = torch.zeros((T, N, venv.n_act), device=dev)
    rew_buf = torch.zeros((T, N), device=dev)
    done_buf = torch.zeros((T, N), device=dev)
    val_buf = torch.zeros((T, N), device=dev)

    log_path = os.path.join(run_dir, "log.csv")
    log_file = open(log_path, "a", newline="")
    logger = csv.writer(log_file)
    if log_file.tell() == 0:
        logger.writerow(["iter", "global_step", "sps", "ep_return", "ep_length",
                         "pi_loss", "v_loss", "entropy", "kl", "lr"])

    obs_raw = venv.reset()
    global_step = start_step
    it = 0
    ep_returns: list[float] = []
    ep_lengths: list[float] = []
    gamma = agent.cfg.gamma

    print(f"training on {args.device} | {N} envs x {T} steps per iter "
          f"| obs {venv.n_obs} act {venv.n_act} | model {args.model}")

    try:
        while global_step < args.total_steps:
            t0 = time.time()
            for t in range(T):
                obs_n, action, logp, value, mu, sigma = agent.act(obs_raw)
                acts_np = action.cpu().numpy()
                next_obs, rewards, terminated, truncated, final_obs, ep_infos = \
                    venv.step(acts_np)

                # bootstrap value for episodes cut off by the time limit
                if final_obs:
                    trunc_ids = [i for i in final_obs if truncated[i]]
                    if trunc_ids:
                        fo = np.stack([final_obs[i] for i in trunc_ids])
                        boot = agent.value_of(fo).cpu().numpy()
                        for j, i in enumerate(trunc_ids):
                            rewards[i] += gamma * boot[j]

                obs_buf[t] = obs_n
                act_buf[t] = action
                logp_buf[t] = logp
                mu_buf[t] = mu
                sigma_buf[t] = sigma
                val_buf[t] = value
                rew_buf[t] = torch.from_numpy(rewards).to(dev)
                done_buf[t] = torch.from_numpy(
                    (terminated | truncated).astype(np.float32)).to(dev)

                for info in ep_infos:
                    ep_returns.append(info["return"])
                    ep_lengths.append(info["length"])
                obs_raw = next_obs
                global_step += N

            # GAE
            with torch.no_grad():
                last_val = agent.value_of(obs_raw)
                adv = torch.zeros_like(rew_buf)
                gae = torch.zeros(N, device=dev)
                lam = agent.cfg.gae_lambda
                for t in reversed(range(T)):
                    next_val = last_val if t == T - 1 else val_buf[t + 1]
                    nonterminal = 1.0 - done_buf[t]
                    delta = rew_buf[t] + gamma * next_val * nonterminal - val_buf[t]
                    gae = delta + gamma * lam * nonterminal * gae
                    adv[t] = gae
                returns = adv + val_buf

            batch = {
                "obs": obs_buf.reshape(-1, venv.n_obs),
                "actions": act_buf.reshape(-1, venv.n_act),
                "logprobs": logp_buf.reshape(-1),
                "mu": mu_buf.reshape(-1, venv.n_act),
                "sigma": sigma_buf.reshape(-1, venv.n_act),
                "advantages": adv.reshape(-1),
                "returns": returns.reshape(-1),
            }
            stats = agent.update(batch)

            it += 1
            sps = int(N * T / (time.time() - t0))
            mean_ret = float(np.mean(ep_returns[-50:])) if ep_returns else 0.0
            mean_len = float(np.mean(ep_lengths[-50:])) if ep_lengths else 0.0
            print(f"it {it:5d} | step {global_step:>10,} | sps {sps:>6,} "
                  f"| ret {mean_ret:7.2f} | len {mean_len:6.0f} "
                  f"| kl {stats['kl']:.4f} | lr {stats['lr']:.1e}")
            logger.writerow([it, global_step, sps, f"{mean_ret:.3f}", f"{mean_len:.1f}",
                             f"{stats['pi_loss']:.4f}", f"{stats['v_loss']:.4f}",
                             f"{stats['entropy']:.4f}", f"{stats['kl']:.5f}",
                             f"{stats['lr']:.2e}"])
            log_file.flush()

            if it % args.ckpt_every == 0:
                agent.save(os.path.join(run_dir, f"ckpt_{it:06d}.pt"),
                           {"global_step": global_step})
            agent.save(os.path.join(run_dir, "ckpt_latest.pt"),
                       {"global_step": global_step})
    except KeyboardInterrupt:
        print("\ninterrupted, saving checkpoint...")
    finally:
        agent.save(os.path.join(run_dir, "ckpt_latest.pt"), {"global_step": global_step})
        log_file.close()
        venv.close()
        print(f"saved {run_dir}/ckpt_latest.pt at step {global_step:,}")


if __name__ == "__main__":
    main()
