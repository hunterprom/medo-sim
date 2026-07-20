# medo-sim

A lightweight "Isaac Sim, but it fits on your machine" for training **Medo**,
the robot dog. MuJoCo physics + parallel environments + PPO, all local, no
30 GB Omniverse install: the whole stack is ~700 MB and trains a walking
policy on a Ryzen 5600 in hours, not days.

## Layout

```
assets/medo.xml         placeholder quadruped (~2.8 kg hobby dog) - swap for real Medo
medo_sim/env.py         locomotion task: velocity commands, rewards, domain rand
medo_sim/vec_env.py     N parallel envs on threads (MuJoCo releases the GIL)
medo_sim/ppo.py         PPO + adaptive KL lr + obs normalization
train.py                training loop -> runs/<name>/ckpt_*.pt + log.csv
play.py                 drive a trained policy with the keyboard in the viewer
view.py                 inspect any model, no policy needed
tools/convert_model.py  Fusion 360 URDF -> training-ready MJCF
docs/EXPORT_GUIDE.md    how to get Medo out of Fusion 360
```

## Quickstart

```
.venv\Scripts\activate
python view.py                          # look at the (placeholder) dog
python train.py --run-name first_walk   # train; Ctrl+C saves and exits
python play.py runs\first_walk\ckpt_latest.pt   # drive it with arrow keys
```

Training prints one line per iteration; `ret` (mean episode return) and `len`
(mean episode length, max 1000) are the health signals. Expect `len` to climb
first, then `ret`. A decent flat-ground gait typically appears somewhere in
the 10–30 M step range (a few hours on the 5600; throughput rises as the
policy stops falling, since resets get rarer).

## Getting the real Medo in

Follow `docs/EXPORT_GUIDE.md` (Fusion 360 -> ACDC4Robot -> URDF ->
`tools/convert_model.py`). Then pass `--model assets/medo_real.xml` to
`train.py` / `play.py` / `view.py`.

## Knobs that matter

- `EnvConfig` in `medo_sim/env.py`: command ranges, reward scales, domain
  randomization. The reward structure is legged-gym-style; if the gait is
  jittery, strengthen `rew_action_rate`; if it drags its feet, raise
  `rew_air_time`.
- `--num-envs`: default 12 (threads). More helps until the CPU saturates.
- `PPOConfig` in `medo_sim/ppo.py`: network sizes, KL target, etc.

## GPU training

The venv has the CUDA build of torch; `train.py` picks the GPU automatically
when one is visible (`--device cpu` forces CPU). Training is deliberately
hybrid: rollout inference runs on a CPU copy of the policy (at batch-of-12
the GPU transfer latency costs more than it saves), while PPO updates run on
the GPU with the whole batch shipped over once per iteration. To reclaim
~6 GB of disk, swap back to the CPU build:

```
.venv\Scripts\python -m pip uninstall -y torch
.venv\Scripts\python -m pip install --no-cache-dir torch
```

## Tests

```
.venv\Scripts\python -m pytest tests -q
```
