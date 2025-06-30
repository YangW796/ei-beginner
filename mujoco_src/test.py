import time
from env import GraspEnv
import glfw

if not glfw.init():
    raise Exception("GLFW could not be initialized!")

env=GraspEnv("./model/UR5+gripper/UR5gripper_2_finger.xml")
# env.robot.move_ee([-0.15,-0.45,0.94])
nearest_pixel, distance = env.camera.get_nearest_point_(show=True)
pos_w=env.camera.pixel_2_world(nearest_pixel[0],nearest_pixel[1],distance)
print("pos_W",pos_w)
env.robot.move_and_grasp(pos_w)
glfw.terminate()