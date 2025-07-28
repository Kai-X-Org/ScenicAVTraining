import os
import random
from math import pi
root_user = os.path.expanduser("~")

param map = localPath(root_user + '/ScenicGymClean/assets/maps/CARLA/Town04.xodr')
param carla_map = 'Town04'
param time_step = 1.0/10
param camera_position = (311, 255, 0)
model scenic.domains.driving.model


success_reward = 10.0
driving_reward = 1.0
crash_penalty = -5.0
out_of_road_penalty = -5.0
speed_reward = 0.1

ego_dir = 180
car_dir = -90
ego_x = 312
car_y = 247

param ego_y = Range(255, 265)
param car_x = Range(290, 306)

ego = new Car on (ego_x, globalParameters.ego_y, 0), facing ego_dir deg,
                                with name "agent0",
                                with goal (311, 235, 0),

car = new Car on (globalParameters.car_x, car_y, 0), facing car_dir deg, 
                                with name "agent1",
                                with goal (320, 246, 0),

monitor Reward():
    done = False
    ego_success = False
    car_success = False

    last_long_1 = ego.position[1]
    last_long_2 = car.position[0]

    drive_dir_1 = ego_dir * pi/180
    drive_dir_2 = car_dir * pi/180

    while True:

        ego.zero_reward()
        car.zero_reward()
        
        current_long_1 = ego.position[1] 
        current_long_2 = car.position[0] 

        ego.add_reward(driving_reward * (-1) * (current_long_1 - last_long_1)) # times -1 since ego is going in -y direction
        car.add_reward(driving_reward * (current_long_2 - last_long_2))
        
        last_long_1 = current_long_1
        last_long_2 = current_long_2

        ego.add_reward(speed_reward * ego.speed_km_h/ego.max_speed_km_h) 
        car.add_reward(speed_reward * car.speed_km_h/car.max_speed_km_h)
        
        angle_rescale = lambda angle: angle + 2 * pi if angle < 0 else angle 

        ego.add_reward(-abs(angle_rescale(ego.yaw) - angle_rescale(drive_dir_1))/pi * 0.1)
        car.add_reward(-abs(angle_rescale(car.yaw) - angle_rescale(drive_dir_2))/pi * 0.1)
        
        if (distance from ego to ego.goal) < 1:
            ego_success = True
            ego.add_reward(success_reward)

        if (distance from car to car.goal) < 1:
            car_success = True
            car.add_reward(success_reward)

        if ego_success and car_success:
            done = True
        
        # can do this since the cars can only crash with each other if they do crash
        if ego.metaDriveActor.crash_vehicle or car.metaDriveActor.crash_vehicle:
            ego.add_reward(crash_penalty)
            car.add_reward(crash_penalty)
            done = True
        
        ego.episode_return += ego.reward
        car.episode_return += car.reward

        if done:
            terminate
        wait
   

require monitor Reward()
record final ego.episode_return as agent0_return
record final ego.episode_return as agent1_return
