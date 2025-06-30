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

log_dir = "./mujoco_src/logs/monitor_3"
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
plt.savefig("./mujoco_src/logs/training_curve.jpg")
plt.show()
plt.close()

def stay(duration):
        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:
            elapsed = (time.time() - starting_time) * 1000
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import GraspEnv
import numpy as np
import time

# ================== 特征提取器 (和训练一致) ==================
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.proprio_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )
        
    def forward(self, observations):
        proprio = torch.cat([
            observations["eef_pos"],
            # observations["target_rel"],  # 如果训练时用了 target_rel，取消注释
        ], dim=1)
        return self.proprio_fc(proprio)

# ==== 创建 DummyVecEnv 包裹的单环境 ====
def make_test_env():
    def _init():
        env = GraspEnv("./model/UR5+gripper/UR5gripper_2_finger.xml", max_episode_steps=1024)
        return Monitor(env)
    return DummyVecEnv([_init])

if __name__ == "__main__":
    # ==== 模型路径 ====
    model_path = "./mujoco_src/logs/best_model/rl_model_71680_steps.zip"  # 请根据你保存的模型文件名调整

    # ==== 创建测试环境 ====
    env = make_test_env()

    # ==== 和训练一致的 policy 参数 ====
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        "activation_fn": torch.nn.ReLU
    }

    # ==== 加载模型 ====
    model = PPO.load(model_path, env=env, custom_objects={"policy_kwargs": policy_kwargs})

    # ==== 开始测试 ====
    num_episodes = 10
    success_count = 0

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]  # DummyVecEnv 返回的是批量（长度为1）

            # 可视化仿真过程（需 GUI 支持）
            env.render()
            time.sleep(0.02)  # 减慢播放速度便于观看

            step += 1

        # 从 info 中提取是否成功
        is_success = info[0].get("is_success", False)
        success_count += int(is_success)
        print(f"Episode {ep + 1} | Reward: {total_reward:.2f} | Steps: {step} | Success: {is_success}")

    env.close()
    del env

    # ==== 统计成功率 ====
    print(f"\n✅ {success_count}/{num_episodes} successful episodes")
    print(f"✅ Success Rate: {success_count / num_episodes:.2%}")