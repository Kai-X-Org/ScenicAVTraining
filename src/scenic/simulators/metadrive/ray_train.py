import gymnasium as gym
from scenic.gym import ScenicGymEnv

def make_env() -> callable:
    def thunk() -> gym.Env:
        ... # create the envioronments

        agents = ['agent0', 'agent1']

        root_user = os.path.expanduser("~")
        sumo_map = root_user + "/ScenicGymClean/assets/maps/CARLA/Town04.net.xml"
        obs_space_dict = {"agent0" :  gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32),
                         "agent1": gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32)}
        # print(f"local decpared obs shape: {obs_space_dict['agent0'].shape}")
        action_space_dict = {'agent0': gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
                             'agent1': gym.spaces.Box(-1.0, 1.0, (2,), np.float32)}

        scenic_file = "exp_uniform.scenic"
        # print("Making Scenario")
       
        scenario = scenic.scenarioFromFile(scenic_file,
                                       model="scenic.simulators.metadrive.model",
                                   mode2D=True)
        
        # print("Made Scenario")
        env = ScenicZooEnv(scenario, 
                           MetaDriveSimulator(sumo_map=sumo_map, render=False, real_time=False),
                           None, 
                           max_steps=50, 
                           observation_space = obs_space_dict, 
                           action_space = action_space_dict, 
                           agents=agents)

        env = ScenicGymEnv(...)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

