from stable_baselines3 import PPO

from env import GraspEnv

env = GraspEnv(model_path="./model/UR5+gripper/UR5gripper_2_finger.xml")
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)