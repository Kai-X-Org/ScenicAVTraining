import magnum as mn
import numpy as np
model scenic.simulators.habitat.model
from scenic.simulators.habitat.actions import *
from scenic.simulators.habitat.behaviors import *
from scenic.simulators.habitat.model import *
from scenic.simulators.habitat.utils import scenic_to_habitat_map
from scenic.core.vectors import Vector
# import math
# import time

goal_point = (-3.2, -1.0, 0)
goal_region = CircularRegion(goal_point, 0.7)

spot = new SpotRobot at (-3.2, -5.0, 0), with yaw 90 deg, with behavior GoRel(y=1.0), with is_learning_agent True
fetch = new FetchRobot at (-4.7, -2.5, 0), with yaw 0 deg, with behavior Traverse(1.0, 0, 0)


monitor Reward():
    done = False
    last_position = spot.position
    last_distance_from_goal = (goal_point - last_position).norm()

    while True:
        if done: # ...need to figure out where this goes...
            terminate
        spot.reward = 0

        distance_from_goal = (distance from spot to goal_point)
        d_distance = distance_from_goal - last_distance_from_goal
        spot.reward += d_distance
        
        if (spot intersects goal_region):
            spot.reward += 10
            done = True

        if (spot intersects fetch):
            spot.reward -= 10
            done = True

        print(spot.reward) 
        print(distance_from_goal)
        wait


require monitor Reward()
