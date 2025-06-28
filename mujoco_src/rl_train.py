from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv  # 多进程并行环境
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import shutil
import torch
import torch.nn as nn
from env import GraspEnv

# ================== 特征提取器 ==================
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
            # observations["target_rel"],
        ], dim=1)
        return self.proprio_fc(proprio)
if __name__ == "__main__":
    # ================== 并行环境数 ==================
    n_envs = 4
    n_steps = 256  # 每个环境收集的步数
    total_timesteps = n_steps *n_envs* 80

    # ================== 并行环境封装 ==================
    def make_env(i):
        log_dir = f"./mujoco_src/logs/monitor_{i}"
        os.makedirs(log_dir, exist_ok=True)
        return lambda: Monitor(GraspEnv("./model/UR5+gripper/UR5gripper_2_finger.xml", max_episode_steps=n_steps), log_dir)

    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # ================== 模型配置 ==================
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        "activation_fn": torch.nn.ReLU
    }

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=n_steps * n_envs,  # 确保整除
        learning_rate=0.01,
        verbose=1,
        tensorboard_log="./mujoco_src/logs/tensorboard/"
    )

    # ================== 回调 & 日志清理 ==================
    checkpoint_cb = CheckpointCallback(
        save_freq=n_steps,
        save_path="./mujoco_src/logs/best_model/"
    )

    # 清理旧模型和日志
    for folder_path in ['./mujoco_src/logs/best_model', './mujoco_src/logs/tensorboard']:
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    # ================== 训练 ==================
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True
    )
