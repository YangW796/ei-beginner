import imageio
from stable_baselines3.common.results_plotter import load_results
from tqdm import trange
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GraspEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from env import GraspEnv
import os
import time

log_dir = "./mujoco_src/logs/monitor"
data = load_results(log_dir)

# 画图
import matplotlib.pyplot as plt
import numpy as np

timesteps = data['t']
rewards = data['r']

plt.figure(figsize=(10, 6))
plt.plot(timesteps, rewards, label="Reward")
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("UR5 Grasping PPO (Gym)")
plt.grid()
plt.legend()
plt.savefig("./mujoco_src/logs/training_curve.png")
plt.close()


# 加载最新模型
# best_model_dir = './mujoco_src/logs/best_model'
# model_files = [f for f in os.listdir(best_model_dir) if f.endswith('.zip')]
# model_files.sort()
# latest_model_path = os.path.join(best_model_dir, model_files[-1])
# model = PPO.load(latest_model_path)

# # 构建环境
# env = DummyVecEnv([
#     lambda: Monitor(GraspEnv("./model/UR5+gripper/UR5gripper_2_finger.xml", 500))
# ])
# env = VecFrameStack(env, n_stack=1)

# # 运行演示
# obs = env.reset()
# for _ in range(500):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     time.sleep(0.1)
#     if done.any():
#         obs = env.reset()
