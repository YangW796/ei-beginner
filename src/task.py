import numpy as np
import pybullet as p
from env import GraspTaskEnv

GRIPPER_MOVING_HEIGHT = 1.15
SIMULATION_STEP_DELAY = 1 / 240.
GRASP_POINT_OFFSET_Z = 1.231 - 1.1
GRASP_SUCCESS_REWARD = 1
GRASP_FAIL_REWARD = -0.3
GRIPPER_GRASPED_LIFT_HEIGHT = 1.2

def task_step(env:GraspTaskEnv,position: tuple, angle: float):
    roll, pitch = 0, np.pi / 2
    x, y, z = position
    yaw = angle
    orn = p.getQuaternionFromEuler([roll, pitch, yaw])
    
    # The return value of the step() method
    observation, reward, done, info = None, 0.0, False, dict()
    grasp_success = False
    env.robot.reset_robot()
    env.robot.move_ee((x, y, GRIPPER_MOVING_HEIGHT, orn))
    
    env.robot.open_gripper()
    env.robot.move_ee((x, y, z +GRASP_POINT_OFFSET_Z, orn),
                    custom_velocity=0.05, max_step=1000)
    item_in_gripper = env.robot.close_gripper(check_contact=True)    
    env.robot.move_ee((x, y, z + GRASP_POINT_OFFSET_Z + 0.1, orn), try_close_gripper=False,
                    custom_velocity=0.05, max_step=1000)
    if item_in_gripper:
        print('Item in Gripper!')
        reward += GRASP_SUCCESS_REWARD
        grasp_success = True

    env.robot.move_ee((x, y, GRIPPER_GRASPED_LIFT_HEIGHT, orn), try_close_gripper=False, max_step=1000)

    env.robot.move_away_arm()
    env.robot.open_gripper()
    rgb, depth, seg = env.camera.shot()
    
    if not grasp_success:
        reward += GRASP_FAIL_REWARD
    
    observation = (rgb, depth, seg)
    env.prev_observation = observation
    
    return observation, reward, done, info
        
        
        
        
        
        
        
        