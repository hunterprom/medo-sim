# Getting Medo out of Fusion 360 and into the sim

> **Status: done (2026-07-24).** The export landed as
> `assets/robot_dog_description` (URDF + STL meshes) and
> `tools/build_medo_real.py` turned it into `assets/medo_real.xml`, which is
> now the default model everywhere. This guide stays for the next re-export;
> if the design changes in Fusion, re-export the package, replace
> `assets/robot_dog_description`, and rerun the builder (it handles the
> steel-placeholder masses and non-watertight meshes automatically).

The pipeline is: **Fusion 360 -> URDF (via the free ACDC4Robot add-in) ->
`tools/convert_model.py` -> training-ready MJCF**. The export itself is the
only manual part; everything after is scripted.

## 1. Prepare the Fusion design (most important step)

ACDC4Robot reads the *joints* in your design, so the design must be jointed
properly:

1. Every rigid group of parts (torso, each hip bracket, each upper leg, each
   lower leg) should be **one component** (or a rigidly-grouped component).
   12 moving components + torso is the target for a 3-DOF-per-leg dog.
2. Connect them with **Revolute joints** (Assemble -> Joint -> Revolute) at the
   real servo axes. Delete or suppress any other joint types.
3. Set **joint limits** on each revolute joint (right-click joint -> Edit Joint
   Limits) to the real servo travel.
4. Give components clean names (`torso`, `FR_hip`, `FR_thigh`, `FR_shank`,
   `FR_foot`...). These become link names. **Anything that touches the ground
   should have "foot" in its name** — the sim uses that to know feet from body.
5. Check masses: Fusion computes them from materials. If you never assigned
   materials, do it now (or at least set the torso's mass to reality —
   right-click component -> Properties). Wrong masses = policies that do not
   transfer to the real dog.
6. The base link (torso) must be the **grounded** component.

## 2. Install ACDC4Robot and export

1. Download the add-in: https://github.com/bionicdl-sustech/ACDC4Robot
   (Code -> Download ZIP, unzip anywhere).
2. In Fusion 360: Utilities -> Add-Ins -> Scripts and Add-Ins -> Add-Ins tab ->
   green `+` -> select the unzipped `ACDC4Robot` folder -> Run.
3. In the ACDC4Robot dialog choose **URDF** (not SDF/MJCF — we convert
   ourselves for full control) and **Simulation: MuJoCo** if asked, then export.
4. You get a folder like `Medo_description/` containing `Medo.urdf` and a
   `meshes/` directory of STLs. Copy that whole folder into this project,
   e.g. `D:\medo-sim\export\Medo_description\`.

## 3. Convert to MJCF

```
python tools/convert_model.py export\Medo_description\Medo.urdf --out assets\medo_real.xml
```

The converter adds the floor, lights, position actuators on every hinge, sim
options, and an empty `home` keyframe. It prints every joint it found — check
that the count is 12.

## 4. Hand-tune (once, ~15 minutes)

Open `assets/medo_real.xml` and:

1. **Foot geoms**: make sure the ground-contact geoms have `foot` in their
   `name`. If Fusion exported meshes as feet, consider replacing each foot's
   collision geom with a small sphere (like `assets/medo.xml` does) — spheres
   give far cleaner contacts than mesh collisions.
2. **`home` keyframe**: set qpos to `x y z  qw qx qy qz  <12 joint angles>` —
   base at standing height with quat `1 0 0 0`, joints in a stable crouch.
   Copy the same 12 joint angles into `ctrl`. Find good angles by opening
   `python view.py --model assets/medo_real.xml` and playing with the sliders.
3. **Servo parameters** on the `<position>` actuators:
   - `forcerange`: the servo's real stall torque (Nm), e.g. `-3 3` for ~3 Nm
     bus servos.
   - `kp`: start at 20–30; higher = stiffer tracking.
   - Add `armature="0.01..0.02"` and `damping="0.1"` on joints (via a default
     class) — this models gearbox reflected inertia and is critical for
     realistic (and stable) dynamics.
4. **Verify**: `python view.py --model assets/medo_real.xml` — check total
   mass is right, the pose looks right, nothing explodes.

## 5. Train on the real body

```
python train.py --model assets\medo_real.xml --run-name medo_real_v1
python play.py runs\medo_real_v1\ckpt_latest.pt --model assets\medo_real.xml
```

If training that worked on the placeholder fails on the real model, the usual
suspects are: masses off by 10x (units!), missing joint armature/damping,
feet not named `foot`, or a `home` keyframe that is not statically stable.
