import time
import numpy as np
import pybullet_data
from data import YCBModels
from robot import ArmBase
from util import Camera
import pybullet as p

GRIPPER_MOVING_HEIGHT = 1.15
SIMULATION_STEP_DELAY = 1 / 240.
GRASP_POINT_OFFSET_Z = 1.231 - 1.1
GRASP_SUCCESS_REWARD = 1
GRASP_FAIL_REWARD = -0.3
GRIPPER_GRASPED_LIFT_HEIGHT = 1.2

def step_simulation():
    p.stepSimulation()
    time.sleep( 1 / 240.)  

    
class GraspTaskEnv:
    def __init__(self,models:YCBModels,camera:Camera,robot:ArmBase,vis=False, obj_num=3):
        self.vis = vis
        self.num_objs = obj_num
        self.camera = camera
        self.models=models
        self.robot=robot
        self.planeID = p.loadURDF("plane.urdf")
        self.tablaID = p.loadURDF("/home/wy/Document/eibeginner/model/pybullet_ur5_robotiq/urdf/objects/table.urdf",[0.0, -0.5, 0.8],p.getQuaternionFromEuler([0, 0, 0]),useFixedBase=True)
        self.UR5StandID = p.loadURDF("/home/wy/Document/eibeginner/model/pybullet_ur5_robotiq/urdf/objects/ur5_stand.urdf",[-0.7, -0.36, 0.0],p.getQuaternionFromEuler([0, 0, 0]),useFixedBase=True)
        
        # load obj
        self.models.load_objects(obj_num)
        self.robot.reset_robot()

        # Observation buffer
        self.prev_observation = tuple()
    
  
        
    def reset(self):
        self.robot.reset_robot()
        self.robot.move_away_arm()
        rgb, depth, seg = self.camera.shot()
        self.prev_observation = (rgb, depth, seg)
        return rgb, depth, seg
        
    
   
    
    def step(self, position: tuple, angle: float):
        roll, pitch = 0, np.pi / 2
        x, y, z = position
        yaw = angle
        orn = p.getQuaternionFromEuler([roll, pitch, yaw])
        
        # The return value of the step() method
        observation, reward, done, info = None, 0.0, False, dict()
        grasp_success = False
        self.robot.reset_robot()
        self.robot.move_ee((x, y, GRIPPER_MOVING_HEIGHT, orn))
        
        self.robot.open_gripper()
        self.robot.move_ee((x, y, z +GRASP_POINT_OFFSET_Z, orn),
                        custom_velocity=0.05, max_step=1000)
        item_in_gripper = self.robot.close_gripper(check_contact=True)    
        self.robot.move_ee((x, y, z + GRASP_POINT_OFFSET_Z + 0.1, orn), try_close_gripper=False,
                        custom_velocity=0.05, max_step=1000)
        if item_in_gripper:
            print('Item in Gripper!')
            reward += GRASP_SUCCESS_REWARD
            grasp_success = True

        self.robot.move_ee((x, y, GRIPPER_GRASPED_LIFT_HEIGHT, orn), try_close_gripper=False, max_step=1000)

        self.robot.move_away_arm()
        self.robot.open_gripper()
        rgb, depth, seg = self.camera.shot()
        
        if not grasp_success:
            reward += GRASP_FAIL_REWARD
        
        observation = (rgb, depth, seg)
        self.prev_observation = observation
        
        return observation, reward, done, info
        
        
        
        
        
        
        
        