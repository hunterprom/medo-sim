# medo-sim

A lightweight "Isaac Sim, but it fits on your machine" for training **Medo**,
the robot dog. MuJoCo physics + parallel environments + PPO, all local, no
30 GB Omniverse install: the whole stack is ~700 MB and trains a walking
policy on a Ryzen 5600 in hours, not days.

## Layout

```
assets/medo_real.xml         the real Medo (from the Fusion 360 export) - the default
assets/robot_dog_description Fusion 360 export: URDF + STL meshes
assets/medo.xml              placeholder quadruped kept for comparison
medo_sim/env.py              locomotion task: velocity commands, rewards, domain rand
medo_sim/vec_env.py          N parallel envs on threads (MuJoCo releases the GIL)
medo_sim/ppo.py              PPO + adaptive KL lr + obs normalization
train.py                     training loop -> runs/<name>/ckpt_*.pt + log.csv
play.py                      drive a trained policy with the keyboard in the viewer
eval.py                      pass/fail check: does it walk fwd/back/left/right?
view.py                      inspect any model, no policy needed
tools/build_medo_real.py     rebuilds medo_real.xml from the export
tools/convert_model.py       generic Fusion 360 URDF -> MJCF converter
docs/EXPORT_GUIDE.md         how the Fusion 360 export was made
```

## Quickstart

```
.venv\Scripts\activate
python view.py                          # look at Medo
python play.py policies\medo_walk_v1.pt --cmd 0.4 0 0   # watch the trained walk
python eval.py policies\medo_walk_v1.pt                 # measure its tracking
python train.py --run-name my_run --deploy --gait-clock # train further
```

`policies\medo_walk_v1.pt` is the trained omnidirectional walking policy
(2026-07-25): deploy-layout observations (IMU + gait clock, no lin vel),
walks/strafes/turns with zero falls in eval, trained against 0.75 m/s
shoves. `docs\media\walk_v1_frame*.png` shows it mid-stride.

Everything defaults to the real Medo (`assets/medo_real.xml`); pass
`--model assets/medo.xml` for the old placeholder.

Training prints one line per iteration; `ret` (mean episode return) and `len`
(mean episode length, max 1000) are the health signals. Expect `len` to climb
first, then `ret`. A decent flat-ground gait typically appears somewhere in
the 10–30 M step range (a few hours on the 5600; throughput rises as the
policy stops falling, since resets get rarer).

## The real Medo model

`assets/medo_real.xml` is built from the Fusion 360 export in
`assets/robot_dog_description` by `tools/build_medo_real.py`. The export's
joint axes sit on the real MG996 servo shafts (verified against the meshes
to ~1 mm), but its masses were Fusion "Steel" placeholders (8 kg!), so the
builder recomputes inertia from the meshes: printed parts at an effective
PLA density, 55 g per servo, plus a 220 g battery/electronics block --
1.55 kg total. Servos are modeled honestly (1.3 Nm cap, MG996-like speed
limit). Collision uses hidden primitives; only the orange toe spheres touch
the ground. Numbers worth verifying against the physical dog (then edit the
constants at the top of the builder and rerun it):

- total mass: weigh the dog; tweak `PLA_EFF` / `BATTERY_MASS`
- servo torque: `SERVO_TORQUE` (MG996R at 5-6 V is roughly 1.0-1.3 Nm)
- which end is the head: the builder trusts the export's fr/br labels;
  if forward comes out mirrored, it is a sign flip, not a retrain

If you re-export from Fusion, drop the new package into
`assets/robot_dog_description` and rerun the builder.

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
