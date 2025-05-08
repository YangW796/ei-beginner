
import time
import mujoco
import numpy as np
from env import GraspEnv

if __name__ == "__main__":
    
    env = GraspEnv("./model/UR5+gripper/UR5gripper_2_finger.xml") 
    # env.robot._move_group_to_joint_target("Arm")
    # obs = env.reset()
    
    # with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    #     for _ in range(1000):
    #         if not viewer.is_running():
    #             break
    #         action = 0.1 * np.random.randn(env.model.nu)
    #         obs, reward, done, _ = env.step(action)
    #         print(f"Reward: {reward:.2f}")
    #         viewer.sync()
    #         time.sleep(0.01)


# N_EPISODES = 100
# N_STEPS = 100

# env.print_info()

# for episode in range(1, N_EPISODES + 1):
#     obs = env.reset()
#     for step in range(N_STEPS):
#         print("#################################################################")
#         print(
#             colored("EPISODE {} STEP {}".format(episode, step + 1), color="white", attrs=["bold"])
#         )
#         print("#################################################################")
#         action = env.action_space.sample()
#         # action = [100,100] # multidiscrete
#         # action = 20000 #discrete
#         observation, reward, done, _ = env.step(action, record_grasps=True)
#         # observation, reward, done, _ = env.step(action, record_grasps=True, render=True)

# env.close()

# print("Finished.")
