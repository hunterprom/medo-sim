# robot_dog_description

URDF export of the Fusion 360 "robotic dog" model.

## Contents
- urdf/robot_dog.urdf  - the robot description
- meshes/*.stl         - per-part collision/visual meshes (mm, scaled 0.001 in URDF)

## Structure
- 13 links: base_link + shoulder/thigh/shank x 4 legs (fr, br, fl, bl)
- 12 revolute joints (3 per leg): <leg>_hip (axis Y), <leg>_thigh (axis X), <leg>_knee (axis X)
- Servos are merged into the link they are mounted on (mass + mesh).
- Joint limits default to +/-90 deg; adjust in the <limit> tags.

## Notes
- All parts currently use the Steel material from Fusion, so masses/inertia are
  placeholder-heavy. Re-assign realistic materials (PLA/ABS for printed parts) in
  Fusion and re-export, or edit the <inertial> blocks, for accurate dynamics.
- Verified: at zero joint angles the URDF reproduces the Fusion assembly exactly
  (max mesh placement error 0.0 m).

## View it
- ROS/rviz: put this folder in a catkin/colcon workspace as package
  "robot_dog_description" and run a robot_state_publisher + rviz, or:
    check_urdf urdf/robot_dog.urdf
- PyBullet:
    import pybullet as p, pybullet_data
    p.connect(p.GUI); p.loadURDF("urdf/robot_dog.urdf")
  (mesh package:// paths may need to be made relative for pybullet)
