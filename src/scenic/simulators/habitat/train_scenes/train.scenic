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

spot = new SpotRobot at (-3.2, -5.0, 0), with yaw 90 deg, with behavior GoRel(y=1.0), with is_learning_agent True
fetch = new FetchRobot at (-4.7, -2.5, 0), with yaw 0 deg, with behavior Traverse(1.0, 0, 0)
