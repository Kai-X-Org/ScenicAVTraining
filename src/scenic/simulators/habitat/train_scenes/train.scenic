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

# hand_x_purturb = Range(-0.1, 0.1)
# hand_y_purturb = Range(-0.1, 0.1)
# hand_z_purturb = Range(0, 0.1)

# ego = new female_0 at (-0.5, -4.8, 0), with yaw -90 deg

# bed = RectangularRegion((0.3, -6.0, 0.63), 0, 1.0, 1.0) # final defined bed width

box_region = RectangularRegion((0.12, -5.5, 0.61), 0, 0.2, 0.2)
# box = new GelatinBox on (0.12, -5.5, 0.61)
box = new GelatinBox on box_region
# spot = new SpotRobot at (-0.9, -5.5, 0), with behavior GrabAndNav()
spot = new SpotRobot at (-3.7, -5.0, 0), with yaw 90 deg
# fetch = new FetchRobot at (-3.7, Range(-2.5, -5.0), 0), with yaw 90 deg, with behavior Traverse()
