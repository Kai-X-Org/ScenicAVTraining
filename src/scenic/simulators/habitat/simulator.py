"""Simulator interface for Meta Habitat."""
import logging
import math
import os
import traceback
import warnings
import torch
from pprint import pprint

import habitat_sim
import magnum as mn
import warnings
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
warnings.filterwarnings('ignore')
from habitat_sim.utils.settings import make_cfg
from matplotlib import pyplot as plt
from habitat_sim.utils import viz_utils as vut
from omegaconf import DictConfig
import numpy as np
from habitat.articulated_agents.robots import FetchRobot
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, HeadPanopticSensorConfig
from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig
import habitat.config.default_structured_configs as cfg
from habitat.config.default import get_agent_config
import habitat
from habitat_sim.physics import JointMotorSettings, MotionType
from habitat.articulated_agent_controllers import HumanoidRearrangeController, HumanoidSeqPoseController
from omegaconf import OmegaConf
from habitat.config.default_structured_configs import HumanoidJointActionConfig, HumanoidPickActionConfig
from habitat.tasks.rearrange.actions.actions import HumanoidJointAction

import scenic.core.errors as errors
from scenic.core.simulators import Simulation, SimulationCreationError, Simulator
from scenic.core.vectors import Vector
from scenic.domains.driving.simulators import DrivingSimulation, DrivingSimulator
import scenic.simulators.habitat.utils as utils

if errors.verbosityLevel == 0:  # suppress pygame advertisement at zero verbosity
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
# import pygame

from scenic.core.simulators import SimulationCreationError
from scenic.syntax.veneer import verbosePrint

# TODO: Import Robot-specific library

def get_sensor_dict(obj):
    """
    Note on sensors:
    Sensors will only be in the correct position if they actually fit on the agent
    e.g. spot has no head, so putting a head rgb on it will make the camera spawn
    in a weird position. 
    To check what sensor fits on which kind of robot, look at the robot's agent class
    in the habitat-lab repo
    """
    obj_type = obj.object_type
    if obj_type == "SpotRobot":
        _sim_sensors = { # TODO temporary
            "third_rgb": cfg.ThirdRGBSensorConfig(width=1024, height=1024),
            # "head_rgb": cfg.HeadRGBSensorConfig(#orientation=[0.0, -1.57, 0.0]
                                                # ),
            "articulated_agent_arm_rgb": cfg.ArmRGBSensorConfig(),
            "articulated_agent_jaw_rgb": cfg.JawRGBSensorConfig(),
            # "head_stereo_left_depth"
        }
    else:
        _sim_sensors = { # TODO temporary
            "third_rgb": cfg.ThirdRGBSensorConfig(width=1024, height=1024),
            "head_rgb": cfg.HeadRGBSensorConfig(#orientation=[0.0, -1.57, 0.0]
                                                ),
        }
    # print(f"sensor pos: {_sim_sensors['head_rgb'].position}")
    return _sim_sensors


def get_action_dict(obj):
    obj_type = obj.object_type
    name = obj.name

    if obj_type == 'SpotRobot':
        return {
            name + "_oracle_magic_grasp_action": cfg.ArmActionConfig(type="MagicGraspAction"),
            name + "_base_vel": cfg.BaseVelocityActionConfig(),
            name + "_oracle_coord_action": cfg.OracleNavActionConfig(type="OracleNavCoordinateAction",
                                                                          spawn_max_dist_to_obj=1.0)
        }

    elif obj_type == 'KinematicHumanoid':
        return {
            name + "_humanoid_joint_action": cfg.HumanoidJointActionConfig(),
            name + "_humanoid_navigate_action": cfg.OracleNavActionConfig(type="OracleNavCoordinateAction",
                                                              motion_control="human_joints", # name + "_human_joints"???
                                                              spawn_max_dist_to_obj=1.0),
            name + "_humanoid_pick_obj_id_action": cfg.HumanoidPickActionConfig(type="HumanoidPickObjIdAction")
        }

    elif obj_type == 'FetchRobot':
        return {
            name + "_oracle_magic_grasp_action": cfg.ArmActionConfig(type="MagicGraspAction"),
            name + "_base_vel": cfg.BaseVelocityActionConfig(),
            name + "_oracle_coord_action": cfg.OracleNavActionConfig(type="OracleNavCoordinateAction",
                                                                          spawn_max_dist_to_obj=1.0),
        }

    else:
        return dict()


class HabitatSimCreationError(SimulationCreationError):
    """
    If anything went wrong in setting up the scene, this error is thrown.
    If Scenic is run using the CLI, The current scene terminates, and a new scene is started
    Args:
    String msg: the message to be given
    """

    def __init__(self, msg):
        self.msg = msg
        super().__init__(self.msg)


class HabitatSimRuntimeError(SimulationCreationError):
    """
    If anything went wrong in running the scene, this error is thrown.
    If Scenic is run using the CLI, The current scene terminates, and a new scene is started
    Args:
    String msg: the message to be given
    Exception exc: the exception thrown by other parts of the code that makes us stop scene
    """

    def __init__(self, msg, exc):
        self.msg = exc.args[0]
        # self.file_name, self.lineno = exc.filename, exc.lineno
        super().__init__(self.msg)
        self.with_traceback(exc.__traceback__)


class HabitatSimulator(Simulator):
    """Implementation of `Simulator`."""

    def __init__(
        self,
        map_path="",
        timeout=10,
        render=True,
        record="",
        timestep=0.1,
    ):
        super().__init__()
        verbosePrint(f"Connecting to Habitat simulator")
        self.timestep = timestep
        self.render = (
            render  # visualization mode ON/OFF, for future use with fast physics
        )
        self.record = record
        self.scenario_number = 0

        # TODO Decide the form of your client
        self.client = None 
        self.env = None
        self.sim = None
        self.agent_dict = dict()
        self.habitat_agents = list()
        self.object_counter = 0

    def createSimulation(self, scene, timestep, **kwargs):
        if timestep is not None and timestep != self.timestep:
            raise RuntimeError(
                "cannot customize timestep for individual Habitat simulations; "
                "set timestep when creating the HabitatSimulator instead"
            )
        if not self.client:
            self.setup_env(scene)
        else:
            self.scene_reset(scene)

        self.scenario_number += 1

        return HabitatSimulation(
            scene,
            self.client,
            self.render,
            self.record,
            timestep=self.timestep,
            scenario_number=self.scenario_number,
            **kwargs,
        )

    def destroy(self):
        # TODO add code to be run when Scenic runs terminates, if needed
        super().destroy()

    def setup_env(self, scene):
        # self.agent_dict = dict()
        # self.sim = None
        # self.habitat_agents = list()
        # self.object_counter = 0 # needed to keep track of ojbect id's

        agent_count = 0
        agent_names = []
        
        # getting the agent configs
        action_dict = dict()
        lab_sensor_dict = dict()
        for obj in scene.objects:
            if obj.is_agent:
                print("Setting up agent: ", obj.object_type)
                self.habitat_agents.append(obj)
                obj._agent_id = agent_count
                print(f"obj agent id: {obj._agent_id}")

                if not obj._only_agent:
                    obj.name = 'agent_' + str(agent_count)
                else: 
                    obj.name = ""

                print("Object name:", obj.name)
                agent_count += 1

                sim_sensors = get_sensor_dict(obj)

                x, y, z = obj.position
                ik_arm_urdf = ""
                if hasattr(obj, '_ik_arm_urdf'):
                    ik_arm_urdf = obj._ik_arm_urdf

                agent_config = utils.create_agent_config(obj.name, obj._articulated_agent_type, obj.urdf_path, 
                                    motion_data_path=obj._motion_data_path, sim_sensors=sim_sensors, ik_arm_urdf=ik_arm_urdf)

                if obj.name in agent_names:
                    raise HabitatSimCreationError(f"Error: two agents have the same name: {obj.name}")
                else:
                    self.agent_dict[obj.name] = agent_config 

                action_dict.update(get_action_dict(obj))
                lab_sensor_dict.update(obj._lab_sensors)
        
        # FIXME line below may cause problem. Defaulting dicts to dict(), but that might not be the default
        # of all the Configs???
        self.action_dict = action_dict
        self.env = utils.init_rearrange_env(self.agent_dict, action_dict, lab_sensor_dict, timestep=self.timestep) 
        self.sim = self.env.sim

        utils.add_scene_camera(self.env, agent_id=None)        

        # utils.add_scene_camera(self.env, name='scene_camera_rgb_2', 
                               # camera_pos=mn.Vector3(2.0, 1.0, 6.5),
                               # orientation=mn.Vector3(mn.Vector3(0, +1.57, 0)), agent_id=None)
        # scenic z = habitat y, scenic x = habitat z, scenic y = habitat x
        # increasing x moves it to the window in bedroom
        # increasing z moves it to the back of the bedroom wall
        # changing x angle moves camera up and down

        utils.add_scene_camera(self.env, name='scene_camera_rgb_2', 
                               camera_pos=mn.Vector3(2.0, 1.9, 0.5),
                               orientation=mn.Vector3(mn.Vector3(-1.57, 0, 0)), agent_id=None)

        utils.add_scene_camera(self.env, name='scene_camera_rgb_3', 
                               camera_pos=mn.Vector3(0, 0.5, 6.5),
                               orientation=mn.Vector3(mn.Vector3(0, -1.57, 0)), agent_id=None)

        obj_attr_mgr = self.sim.get_object_template_manager()
        prim_attr_mgr = self.sim.get_asset_template_manager()
        stage_attr_mgr = self.sim.get_stage_template_manager()
        rigid_obj_mgr = self.sim.get_rigid_object_manager()
        agents_mgr = self.sim.agents_mgr

        self.client = {
            "env" : self.env,
            "sim" : self.sim,
            "obj_attr_mgr" : obj_attr_mgr,
            "prim_attr_mgr" : prim_attr_mgr,
            "stage_attr_mgr" : stage_attr_mgr,
            "rigid_obj_mgr" : rigid_obj_mgr,
            "agents_mgr" : agents_mgr,
        }

    def scene_reset(self, scene):
        agent_count = 0
        agent_names = []
        
        # getting the agent configs
        action_dict = dict()
        lab_sensor_dict = dict()
        for obj in scene.objects:
            if obj.is_agent:
                obj._agent_id = agent_count

                if not obj._only_agent:
                    obj.name = 'agent_' + str(agent_count)
                else: 
                    obj.name = ""

                agent_count += 1

                if obj.name in agent_names:
                    raise HabitatSimCreationError(f"Error: two agents have the same name: {obj.name}")
        

class HabitatSimulation(Simulation):
    """
    Simulation class for Habitat-Scenic
    """

    def __init__(self, scene, client, render, record, timestep=0.1, scenario_number=0, **kwargs):
        print('initializing!')
        self.client = client # FIXME, fix this
        self.render = True
        self.record = record
        self.timestep = timestep
        if 'cfg' in kwargs:
            self.cfg = kwargs['cfg']

        self.sim = client["sim"]
        self.env = client["env"]
        self.obj_attr_mgr = client["obj_attr_mgr"]
        self.prim_attr_mgr = client["prim_attr_mgr"]
        self.stage_attr_mgr = client["stage_attr_mgr"] 
        self.rigid_obj_mgr = client["rigid_obj_mgr"] 
        self.agents_mgr = client["agents_mgr"] 
        self.actions = None


        self.observations = list()
        self.env_observations = list()
        # self.habitat_agents = list()
        self.scenario_number = scenario_number  # used for naming of videos
        # self.device = torch.device('cuda') 
        self.step_action_dict = {
            "action": tuple(),
            "action_args": dict()
        }
        # self.object_counter = 0 # needed to keep track of ojbect id's

        super().__init__(scene, timestep=timestep, **kwargs)

    def setup(self):
        self.env.reset() 
        super().setup()  # Calls createObjectInSimulator for each object
        return

    def createObjectInSimulator(self, obj):
        """
        Spawns the object in the habitat simulator.
        If the object has a mesh, adds it to the collision_world
        to enable collision avoidance
        Args:
        obj: the scenic object, needs to have a name and position field

        Returns:
        Tuple(bool success, status_message)
        """
        # print(f"CREATING {obj.name}")

        for action_name, action_space in self.env.action_space.items():
            print(action_name, action_space)
        if obj.is_agent:
            # breakpoint()
            # if self.scenario_number == 2:
                # breakpoint()
            # print(f"SCENARIO NUMBER: {self.scenario_number}")
            art_agent = self.env.sim.agents_mgr[obj._agent_id].articulated_agent  

            obj._articulated_agent = art_agent
            if obj._articulated_agent_type == 'KinematicHumanoid':

                # print("CREATING HUMAN")
                # print('data_path:!!!', obj._motion_data_path)

                art_agent.sim_obj.motion_type = MotionType.KINEMATIC 
                art_agent._fixed_base = True  
                obj._humanoid_controller = HumanoidRearrangeController(obj._motion_data_path)

                # HOPEFULLY IT IS NOT THIS STUFF BELOW GIVING THE PROBLEM
                obj._humanoid_joint_action = HumanoidJointAction(config=HumanoidJointActionConfig(),
                                                                 sim=self.sim, name=f'agent_{obj._agent_id}')
            else:
                art_agent._fixed_base = False
                art_agent.sim_obj.motion_type = MotionType.DYNAMIC
                if obj._has_grasp:
                    obj._grasp_manager = self.sim.agents_mgr[obj._agent_id].grasp_mgrs[0]
                
                extra_files = {"net_meta_dict.pkl": ""} # TODO temporary hardcoding

            x, y, z, _, _, _ = self.scenicToHabitatMap((obj.position[0], obj.position[1], obj.position[2],0, 0, 0))
            art_agent.base_pos = mn.Vector3(x, y, z) 
            art_agent.base_rot = obj.yaw - 1.57

        else:
            handle = obj._object_file_handle
            self.obj_attr_mgr.load_configs('/home/kxu/habitat-lab/data/objects/ycb/configs/')
            obj_template_handle = self.obj_attr_mgr.get_template_handles(handle)[0]
            obj._managed_rigid_object = self.rigid_obj_mgr.add_object_by_template_handle(obj_template_handle)
            obj._object_id = obj._managed_rigid_object.object_id

            x, y, z, _, _, _ = self.scenicToHabitatMap((obj.position[0], obj.position[1], obj.position[2],0, 0, 0))
            obj._managed_rigid_object.translation = np.array([x, y, z])
            obj._managed_rigid_object.rotation = mn.Quaternion.rotation(mn.Rad(obj.yaw), [0.0, 1.0, 0.0]) 
            

    def executeActions(self, allActions):
        """
        execute action for each object. Does not immediately render,
        but instead buffers the object
        """
        for agent, actions in allActions.items():
            for action in actions:
                try:
                    # print("OBJECT TYPE:",agent.object_type)
                    # print("OBJECT ACTION:", action)
                    a = action.applyTo(agent, self)
                except Exception as e:
                    print(f"Failed to execute action, exception:\n{str(e)}")
                    logging.error(traceback.format_exc())
        return

    def step(self):
        # appending observations from self.sim.get_sensor_observations() is
        # necessary since env.step(...) does not return observation
        # from sensors added after env is created, for some reason
        # temp_param = self.env.action_space["agent_0_base_velocity_action"].sample()
        # print(f"VELOCITY ACTION PARAM: {temp_param}")
        # this action space is gym.spaces.dict.Dict(agent_0_base_vel:Box(-20.0, 20.0, (2,), float32))
        # might want to make it smaller than this. say +/- 1 for each. and need to figure out conversion
        # to x y z vel from r, theta
        # observation space for the jaw camera is Box(0, 255, (256, 256, 3), uint8)

        if self.actions:
            print(f"Received Actions: {actions}")
            self.step_action_dict["action"] += tuple(["agent_0_base_vel"])
            self.step_action_dict["action_args"]["agent_0_base_vel"] = self.actions 

        self.env_observations.append(self.env.step(self.step_action_dict))
        self.observations.append(self.sim.get_sensor_observations())  # for sim sensors (scene carmeras etc.)
        # breakpoint()
        
        # TODO call articulated_agent.update to update camera angles...wait, might not need it
        self.step_action_dict = {
            "action": tuple(),
            "action_args": dict()
        } # clearing step_action_dict

    def getProperties(self, obj, properties):
        if obj.is_agent:
            if obj.object_type == 'SpotRobot':
                ee_pos = obj._articulated_agent.ee_transform().translation
                x, y, z, _, _, _ = self.habitatToScenicMap((ee_pos[0], ee_pos[1], ee_pos[2], 0, 0, 0))
                # print(f"EE pos{x, y, z}")
                obj.ee_pos = Vector(x, y, z)

            # if obj.object_type == 'KinematicHumanoid':
                # offset =  obj._articulated_agent.base_transformation.transform_vector(mn.Vector3(0, 0.3, 0))
                # hand_pos = obj._articulated_agent.ee_transform(0).translation + offset
                # obj.ee_pos = hand_pos

            x, y, z = obj._articulated_agent.base_pos
            x, y, z, _, _, _ = self.habitatToScenicMap((x, y, z, 0, 0, 0))
            rotation = obj._articulated_agent.base_rot
            
            d = dict(
                    position=Vector(x, y, z),
                    yaw=rotation,
                    pitch=0,
                    roll=0,
                    speed=0,
                    velocity=Vector(0, 0, 0),
                    angularSpeed=0,
                    angularVelocity=Vector(0, 0, 0),
            )
        else:
            x, y, z = obj._managed_rigid_object.translation
            rotation = obj._managed_rigid_object.rotation 
            x, y, z, _, _, _ = self.habitatToScenicMap((x, y, z, 0, 0, 0))
            # print(f"Obj pos: {x, y, z}")
            d = dict(
                    position=Vector(x, y, z),
                    yaw=0,
                    pitch=0,
                    roll=0,
                    speed=0,
                    velocity=Vector(0, 0, 0),
                    angularSpeed=0,
                    angularVelocity=Vector(0, 0, 0),
            )

        return d

    def get_obs(self):
        # use the env_observations rather than self.observations; the latter adds an extra dimension at -1 for some reason
        if len(self.env_observations) == 0:
            return None

        return self.env_observations[-1]['agent_0_articulated_agent_jaw_rgb']
        
    def get_info(self):
        return None

    def get_reward(self):
        # for obj in self.scene.objects:
            # if obj.name == "agent_0":
        return self.scene.objects[0].reward
        
        

    def destroy(self):
        print("FINISH SCENE, DESTROYING...")
        # FIXME probably now need to destroy objects now???
        # self.env.reset()
        # self.env.close()
        # print("closed env")
        # super().destroy()
        # return
        make_vid = False
        if make_vid:
            folder_name = "test_run_vids/"

            vut.make_video(
                self.observations,
                "scene_camera_rgb",
                "color",
                f"/home/kxu/ScenicGymClean/src/scenic/simulators/habitat/{folder_name}scene_overview_{self.scenario_number}",
                open_vid=False,
            )

            vut.make_video(
                self.observations,
                "scene_camera_rgb_2",
                "color",
                f"/home/kxu/ScenicGymClean/src/scenic/simulators/habitat/{folder_name}scene_overview_2_{self.scenario_number}",
                open_vid=False,
            )

            vut.make_video(
                self.observations,
                "scene_camera_rgb_3",
                "color",
                f"/home/kxu/ScenicGymClean/src/scenic/simulators/habitat/{folder_name}scene_overview_3_{self.scenario_number}",
                open_vid=False,
            )

            vut.make_video(
                self.observations,
                "agent_0_third_rgb",
                "color",
                f"/home/kxu/ScenicGymClean/src/scenic/simulators/habitat/{folder_name}test_spot_{self.scenario_number}",
                open_vid=False,
            )

            # vut.make_video(
                # self.observations,
                # "agent_0_head_rgb",
                # "color",
                # f"/home/kxu/ScenicGymClean/src/scenic/simulators/habitat/{folder_name}test_spot_head_{self.scenario_number}",
                # open_vid=False,
            # )
            vut.make_video(
                self.observations,
                "agent_0_articulated_agent_arm_rgb",
                "color",
                f"/home/kxu/ScenicGymClean/src/scenic/simulators/habitat/{folder_name}test_spot_arm_{self.scenario_number}",
                open_vid=False,
            )

            vut.make_video(
                self.observations,
                "agent_0_articulated_agent_jaw_rgb",
                "color",
                f"/home/kxu/ScenicGymClean/src/scenic/simulators/habitat/{folder_name}test_spot_jaw_{self.scenario_number}",
                open_vid=False,
            )

            vut.make_video(
                self.observations,
                "agent_1_head_rgb",
                "color",
                f"/home/kxu/ScenicGymClean/src/scenic/simulators/habitat/{folder_name}test_fetch_head_{self.scenario_number}",
                open_vid=False,
            )


            # vut.make_video(
                # self.observations,
                # "agent_1_third_rgb",
                # "color",
                # f"/home/kxu/ScenicGymClean/src/scenic/simulators/habitat/{folder_name}test_spot_1_{self.scenario_number}",
                # open_vid=False,
            # )

            # vut.make_video(
                # self.observations,
                # "agent_2_third_rgb",
                # "color",
                # f"/home/kxu/ScenicGymClean/src/scenic/simulators/habitat/{folder_name}test_fetch_1_{self.scenario_number}",
                # open_vid=False,
            # )

        super().destroy()
        return

    def habitatToRobotMap(self, pose):
        """
        Converts from the habitat map frame to the Robot map frame
        Args:
        pose = Tuple(x, y, z, yaw)
        """
        pass

    def robotToHabitatMap(self, pose):
        """
        Converts from the Robot map frame to the habitat map frame
        Args:
        pose: (x, y, z, yaw)
        """
        pass

    def scenicToRobotMap(self, pose, obj=None):
        """
        Converts from the Scenic map coordinate to the Robot map frame
        Args:
        pose: (x, y, z, yaw)
        """
        pass


    def robotToScenicMap(self, pose, obj=None):
        """
        Converts from the Robot 'map' frame coordinate to the Scenic map coordinate
        Args:
        pose: (x, y, z, yaw)
        """
        assert len(pose) == 4
        pass

    def scenicToHabitatMap(self, pose, obj=None):
        """
        Converts from the Scenic map coordinate to the habitat map frame coordinate
        Args:
        """
        g = np.array([[0, 1, 0, 0], 
                      [0, 0, 1, 0], 
                      [1, 0, 0, 0], 
                      [0, 0, 0, 1]])
        x, y, z, roll, pitch, yaw = pose
        x, y, z, _ = g @ np.array([x, y, z, 1])

        new_roll = pitch
        new_pitch = yaw
        new_yaw = roll
        return (x, y, z, new_roll, new_pitch, new_yaw)

    def habitatToScenicMap(self, pose, obj=None):
        """
        Converts from the habitat map frame coordinate to the Scenic map coordinate
        Args:
        pose: (x, y, z, roll, pitch, yaw)
        """
        # assert len(pose) == 4
        # return self.RobotToScenicMap(self.HabitatToRobotMap(pose), obj=obj)
        g = np.array([[0, 1, 0, 0], 
                      [0, 0, 1, 0], 
                      [1, 0, 0, 0], 
                      [0, 0, 0, 1]])
        g = np.linalg.inv(g)
        x, y, z, roll, pitch, yaw = pose
        x, y, z, _ = g @ np.array([x, y, z, 1])
        new_roll = yaw
        new_pitch = roll
        new_yaw = pitch
        return (x, y, z, new_roll, new_pitch, new_yaw)
