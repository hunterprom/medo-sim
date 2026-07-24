# Deploying policies to the physical Medo

Status: **bridge ready, calibration pending, IMU needed for real walking.**

```
reference/robotic_dog_walking_cycle.ino  your original hand-coded trot (archived)
medo_serial_bridge/                      new ESP32-S3 firmware: serial -> servos
servo_map.py                             sim joint <-> PCA9685 channel/dir/trim
run_policy.py                            host-side runner (bench mode for now)
```

## The path

1. **Flash the bridge** (`medo_serial_bridge.ino`, Arduino IDE, same
   Adafruit PWM Servo Driver library as your trot sketch). On boot the dog
   moves to the standing pose and waits; if the host stops streaming for
   0.4 s it eases back to standing (watchdog).
2. **Calibrate** (dog on a stand, legs free):
   - `python deploy\run_policy.py <ckpt> --port COMx --action-scale 0.0`
     streams the frozen home pose. Compare the physical stance to
     `python view.py` at the home keyframe.
   - Wrong stance on a joint: edit its `HOME_SERVO_DEG` in `servo_map.py`
     (and the copy in the bridge's `HOME_DEG`).
   - Then nudge one joint at a time (temporarily bump `--action-scale`)
     and flip that joint's `DIR` sign if it moves the wrong way.
   - The current table is an educated guess from your trot sketch's
     trims/dirs, incl. its per-leg fixed abduction angles (35/45/45/75).
3. **Bench test**: `--action-scale 0.25`, dog ON A STAND, press `w` --
   the legs should make coordinated stepping motions.
4. **Real walking needs two things** (not built yet):
   - an IMU (MPU6050 on the same I2C bus works) streamed back through the
     bridge so the policy gets real gravity + angular velocity;
   - a policy trained without the base-linear-velocity observation
     (unobservable on the real dog). `EnvConfig(obs_lin_vel=False)` exists
     for exactly this -- retrain, then wire the IMU numbers into
     `run_policy.py` in place of the open-loop assumptions.

## Findings from your trot sketch worth knowing

- **Link lengths look swapped in the .ino**: it uses `L_HIP=80, L_KNEE=60`
  (mm), but the Fusion CAD meshes measure thigh ~59 mm and shank ~80 mm.
  If the printed dog matches the CAD, your IK trot computes distorted foot
  arcs (masked while standing by the hand trims). Measure hip-to-knee and
  knee-to-toe once with a ruler; whichever is right should fix the other
  artifact (sim model or trot sketch).
- Left/right naming may be mirrored between the export and reality; the
  calibration step catches this (a "left" command strafing right = swap
  the fl/fr and bl/br channel rows in `servo_map.py`).
- MG996R at 6 V is ~1.1 Nm stall; the sim trains against 1.3 Nm with the
  real speed limit, so anything the sim dog does should be within reach of
  the hardware, minus battery sag.
