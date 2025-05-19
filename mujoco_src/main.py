import glfw
import time
import mujoco
import numpy as np
from env import GraspEnv

if __name__ == "__main__":
    if not glfw.init():
        raise Exception("GLFW initialization failed")
    env = GraspEnv("./model/UR5+gripper/UR5gripper_2_finger.xml") 
    
    # for _ in range(1000):
    #     if not env.viewer.is_running():
    #         break
    #     action = 0.1 * np.random.randn(env.model.nu)
    #     obs, reward, done, _ = env.step(action)
    #     time.sleep(0.05)
    
    # glfw.terminate()  
