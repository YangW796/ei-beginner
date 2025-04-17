
import numpy as np
import pybullet as p
import os

import pybullet_data
from data import YCBModels
from env import GraspTaskEnv
from robot import ArmBase
from util import Camera
from task import task_step

def demo():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    ycb_models = YCBModels(
        os.path.join('/home/wy/Document/eibeginner/model/pybullet_ur5_robotiq/data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = Camera((0, -0.5, 1.5), 0.1, 5, (320, 320), 40)
    robot = ArmBase("/home/wy/Document/eibeginner/model/pybullet_ur5_robotiq/urdf/ur5_robotiq_85.urdf")
    env= GraspTaskEnv(ycb_models, camera,robot,obj_num=2, vis=True)
    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
    (rgb, depth, seg) = env.reset()
    step_cnt = 0
    while True:
        h_, w_ = np.unravel_index(depth.argmin(), depth.shape)
        x, y, z = camera.rgbd_2_world(w_, h_, depth[h_, w_])

        p.addUserDebugLine([x, y, 0], [x, y, z], [0, 1, 0])
        p.addUserDebugLine([x, y, z], [x, y, z+0.05], [1, 0, 0])

        (rgb, depth, seg), reward, done, info = task_step(env,(x, y, z), 1)

        print('Step %d, grasp at %.2f,%.2f,%.2f, reward %f, done %s, info %s' %
              (step_cnt, x, y, z, reward, done, info))
        step_cnt += 1
    

if __name__ == '__main__':
    demo()