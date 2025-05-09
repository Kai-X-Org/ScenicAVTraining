# Should there be a reward for total distance travelled foward? To give a denser signal?
# but that's taken care of by the longitudinal reward

import os
import random
from math import pi
root_user = os.path.expanduser("~")
param verifaiSamplerType = 'halton'
param map = localPath(root_user + '/ScenicGym/assets/maps/CARLA/Town04.xodr')
param carla_map = 'Town04'
param time_step = 1.0/10
param camera_position = (311, 255, 0)
model scenic.domains.driving.model

success_reward = 10.0
driving_reward = 1.0
crash_penalty = -5.0
out_of_road_penalty = -5.0
speed_reward = 0.1

car1_dir = 180
car2_dir = -90

param ego_y = VerifaiRange(255, 265)
param car2_x = VerifaiRange(290, 306)

ego = new Car on (312, globalParameters.ego_y, 0), facing car1_dir deg,
                                with name "agent0",
                                with goal (311, 235, 0),

car2 = new Car on (globalParameters.car2_x, 247, 0), facing car2_dir deg, 
                                with name "agent1",
                                with goal (320, 246, 0),

monitor Reward(car1, car2):
    done = False
    car1_success = False
    car2_success = False

    last_long_1 = car1.position[1]
    last_long_2 = car2.position[0]

    drive_dir_1 = car1_dir * pi/180
    drive_dir_2 = car2_dir * pi/180

    while True:

        car1.zero_reward()
        car2.zero_reward()
        
        current_long_1 = car1.position[1] 
        current_long_2 = car2.position[0] 

        car1.add_reward(driving_reward * (-1) * (current_long_1 - last_long_1)) # times -1 since car1 is going in -y direction
        car2.add_reward(driving_reward * (current_long_2 - last_long_2))
        
        # ...Other reward terms...

        if (distance from car1 to car1.goal) < 1:
            car1_success = True
            car1.add_reward(success_reward)

        if (distance from car2 to car2.goal) < 1:
            car2_success = True
            car2.add_reward(success_reward)

        if car1_success and car2_success:
            done = True
        
        if car1.metaDriveActor.crash_vehicle or car2.metaDriveActor.crash_vehicle:
            car1.add_reward(crash_penalty)
            car2.add_reward(crash_penalty)
            done = True

        if done:
            terminate
        wait

require monitor Reward(ego, car2)
