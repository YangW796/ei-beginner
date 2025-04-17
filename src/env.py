import time
import numpy as np
import pybullet_data
from data import YCBModels
from robot import ArmBase
from util import Camera
import pybullet as p



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
        
    
