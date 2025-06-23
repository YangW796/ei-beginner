from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import os
import shutil

import torch
import torch.nn as nn
from env import GraspEnv


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.proprio_fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU()
        )
        
    def forward(self, observations):
        proprio = torch.cat([
            # observations["joint_pos"],
            observations["eef_pos"],
            observations["target_rel"],
        ], dim=1)
        return self.proprio_fc(proprio)

# ========== 环境配置 ==========
n_steps_=200 #每次任务执行500步
log_dir = "./mujoco_src/logs/monitor"
env = DummyVecEnv([
    lambda: Monitor(GraspEnv("./model/UR5+gripper/UR5gripper_2_finger.xml",n_steps_), log_dir)
])
env = VecFrameStack(env, n_stack=1)

# ========== 模型配置 ==========
policy_kwargs = {
    "features_extractor_class": CustomFeatureExtractor,
    "net_arch": dict(pi=[64, 64], vf=[64, 64]),
    "activation_fn": torch.nn.ReLU
}

model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    n_steps=n_steps_,
    batch_size=10,
    learning_rate=0.01,
    verbose=1,
    tensorboard_log="./mujoco_src/logs/tensorboard/"
)

checkpoint_cb = CheckpointCallback(
    save_freq=200,
    save_path="./mujoco_src/logs/best_model/"
)
#=============重置文件夹=============
folder_path = './mujoco_src/logs/best_model'

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
        
folder_path = './mujoco_src/logs/tensorboard'

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
# ========== 训练 ==========
model.learn(
    total_timesteps=20000,
    callback=checkpoint_cb,
    progress_bar=True
)
