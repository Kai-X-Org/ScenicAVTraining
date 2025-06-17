import magnum as mn
import numpy as np
model scenic.simulators.habitat.model
from scenic.simulators.habitat.actions import *
from scenic.simulators.habitat.behaviors import *
from scenic.simulators.habitat.model import *
from scenic.simulators.habitat.utils import scenic_to_habitat_map
from scenic.core.vectors import Vector
import math
import time

behavior SpotPickUp():
    raise_pos = np.array([0.0, -3.14, 0.00, 1.57, 0.0, 0.0, 0.0]) # forarm raise
    do MoveToJointAngles(raise_pos)
    
    raise_pos = [0.0, -1.0, 0.0, 1.57, 0.0, 0.0, 0.0] # shoulder raise
    do MoveToJointAngles(raise_pos)
    
    # # spot_ee_pos = self.ee_pos
    # box_pos = box.position
    # # diff_pos = box_pos - spot_ee_pos
    # diff_pos = box_pos - self.ee_pos
    # diff_norm = np.linalg.norm(np.array([diff_pos[0], diff_pos[1], diff_pos[2]]))  
    # print(f"pos_difference norm: {diff_norm}")
    # if diff_norm < 0.25:
        # take SnapToObjectAction(box)
    # else:
        # terminate

    # raise_pos = np.array([0.0, -3.14, 0.00, 3.14, 0.0, 0.0, 0.0]) # retract_arm
    # do MoveToJointAngles(raise_pos, steps=50)

behavior MoveToJointAngles(joint_angles, steps=50):
    start_pos = np.array(self._articulated_agent.arm_joint_pos)
    delta_pos = (joint_angles - start_pos)/steps
    for _ in range(steps):
        new_pos = list(start_pos + delta_pos)
        take SpotMoveArmAction(arm_ctrl_angles=new_pos)
        start_pos = np.array(self._articulated_agent.arm_joint_pos)


box_region = RectangularRegion((0.12, -5.5, 0.61), 0, 0.2, 0.2)
# box = new GelatinBox on (0.12, -5.5, 0.61)
box = new GelatinBox on box_region
spot = new SpotRobot at (-0.9, -5.5, 0), with behavior SpotPickUp
fetch = new FetchRobot at (-3.7, -2.5, 0), with yaw 0 deg
# spot = new SpotRobot at (-0.9, -5.5, 0), with behavior GrabAndNav()
