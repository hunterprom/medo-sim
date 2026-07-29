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


def _yquat(angle: float) -> np.ndarray:
    """Quaternion (wxyz) for a rotation about the world y axis."""
    return np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])


def _xquat(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])


def slope_boxes(difficulty: float, rng: np.random.Generator) -> list:
    """One large tilted box: an exact analytic slope (no hfield sink).

    The top face passes through z=0 at the origin, so the dog spawns
    directly on the incline. Returns [(pos, size, quat_wxyz), ...].
    """
    d = float(np.clip(difficulty, 0.0, 1.0))
    ang = np.radians(4.0 + 10.0 * d) * (1 if rng.random() < 0.5 else -1)
    along_x = rng.random() < 0.7
    size = np.array([4.0, 4.0, 0.25])
    if along_x:
        quat = _yquat(-ang)
        local_z = np.array([-np.sin(ang), 0.0, np.cos(ang)])
    else:
        quat = _xquat(ang)
        local_z = np.array([0.0, -np.sin(ang), np.cos(ang)])
    # sink the box along its own z so the TOP face plane contains the origin
    pos = -size[2] * local_z
    return [(pos, size, quat)]


def rough_boxes(difficulty: float, rng: np.random.Generator,
                pool: int = 32) -> list:
    """Random-height tile field: exact box contacts instead of the hfield,
    whose sphere collisions let feet sink into bumps by up to their radius
    (the clipping visible in the user's screenshot)."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    amp = 0.008 + 0.030 * d          # tile top spread 0.8 -> 3.8 cm
    cell = 0.8
    nx, ny = 8, 4                    # 8 x 4 tiles = 6.4 x 3.2 m, pool of 32
    boxes = []
    x0, y0 = -(nx - 1) * cell / 2, -(ny - 1) * cell / 2
    for i in range(nx):
        for j in range(ny):
            x, y = x0 + i * cell, y0 + j * cell
            if abs(x) < SPAWN_R + 0.15 and abs(y) < SPAWN_R + 0.15:
                h = 0.012            # level spawn tiles
            else:
                h = 0.004 + rng.uniform(0.0, amp)
            # bottom 5 mm below the floor: a bottom face exactly at z=0
            # z-fights the floor plane into flickering blotches
            half = (h + 0.005) / 2
            boxes.append((np.array([x, y, h - half]),
                          np.array([cell / 2 + 0.001, cell / 2 + 0.001, half]),
                          np.array([1.0, 0.0, 0.0, 0.0])))
    return boxes[:pool]


def stair_boxes(difficulty: float, rng: np.random.Generator,
                max_steps_per_side: int = 16) -> list:
    """Pyramid stairs from exact box primitives, climbing away from spawn."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    rise = 0.02 + 0.05 * d
    run = 0.14 + rng.uniform(-0.02, 0.04)
    boxes = []
    for k in range(1, max_steps_per_side + 1):
        top = k * rise
        # top unchanged, bottom 5 mm below the floor (coplanar bottom faces
        # z-fight the floor plane into flickering blotches)
        half_h = (top + 0.005) / 2
        if k == max_steps_per_side:
            # summit platform: without it the staircase ends in a cliff, the
            # climber walks off the far edge and every stairs episode ends
            # in a "fall" -- the curriculum could never mark stairs passed
            half_x = 0.6
            x = SPAWN_R + (k - 1) * run + half_x
        else:
            half_x = run / 2 + 0.003   # slight overlap hides hairline seams
            x = SPAWN_R + (k - 0.5) * run
        size = np.array([half_x, 3.0, half_h])
        for sx in (+1, -1):
            boxes.append((np.array([sx * x, 0.0, top - half_h]), size.copy(),
                          np.array([1.0, 0.0, 0.0, 0.0])))
    return boxes
