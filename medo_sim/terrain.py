"""Procedural terrain heightfields for blind all-terrain training.

Each MedoEnv owns its own MjModel, so a per-episode terrain swap is just a
numpy write into model.hfield_data -- no recompiling, no cross-env locking.
Heights are generated in meters; the env normalizes them into the hfield's
[0, 1] data range and stores the true scale in hfield_size[2].
"""

from __future__ import annotations

import numpy as np

TERRAIN_KINDS = ("flat", "rough", "slope", "stairs")
SPAWN_R = 0.35        # flat spawn disc radius (m); every kind starts level
SPAWN_BLEND = 0.25    # blend band beyond the disc


def _smooth(a: np.ndarray, passes: int = 2) -> np.ndarray:
    for _ in range(passes):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
             + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
    return a


def make_heights(kind: str, difficulty: float, n: int, extent: float,
                 rng: np.random.Generator) -> np.ndarray:
    """(n, n) surface heights in meters; rows follow y, columns follow x."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    xs = np.linspace(-extent / 2, extent / 2, n)
    X = np.tile(xs, (n, 1))          # varies along columns = world x
    Y = X.T
    if kind == "flat":
        return np.zeros((n, n))
    if kind == "rough":
        amp = 0.005 + 0.030 * d      # 0.5 -> 3.5 cm bumps
        h = _smooth(rng.uniform(0.0, amp, (n, n)), passes=2)
    elif kind == "slope":
        grade = np.tan(np.radians(4.0 + 10.0 * d))   # 4 -> 14 deg
        axis = X if rng.random() < 0.7 else Y        # mostly along the walk
        h = (axis if rng.random() < 0.5 else -axis) * grade
    elif kind == "stairs":
        rise = 0.02 + 0.05 * d       # 2 -> 7 cm per step
        run = 0.14 + rng.uniform(-0.02, 0.04)
        # pyramid stairs climbing away from the spawn disc in both x
        # directions: forward commands climb, backward commands descend
        h = rise * np.floor(np.clip(np.abs(X) - SPAWN_R, 0.0, None) / run)
    else:
        raise ValueError(f"unknown terrain kind {kind!r}")
    # flatten + blend a spawn disc so the dog always starts on level ground
    R = np.hypot(X, Y)
    blend = np.clip((R - SPAWN_R) / SPAWN_BLEND, 0.0, 1.0)
    ctr = h[n // 2, n // 2]
    h = ctr * (1.0 - blend) + h * blend
    h -= h.min()
    return h
