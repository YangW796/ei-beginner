from env import GraspEnv
import glfw

if not glfw.init():
    raise Exception("GLFW could not be initialized!")

env=GraspEnv("./model/UR5+gripper/UR5gripper_2_finger.xml")
env.robot.move_ee([0.0,-0.6,0.96])
