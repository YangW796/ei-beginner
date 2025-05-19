from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

from env import GraspEnv
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        
        # 本体特征处理
        self.proprio_fc = nn.Sequential(
            # nn.Linear(6 + 6 + 3 + 4 + 1 + 3 + 6, 128),
            nn.Linear(12, 64),
            nn.ReLU()
        )
        
    def forward(self, observations):
        
        # 处理本体输入
        proprio = torch.cat([
            observations["joint_pos"],
            # observations["proprioception"]["joint_vel"],
            observations["eef_pos"],
            # observations["proprioception"]["eef_quat"],
            # observations["proprioception"]["gripper"],
            observations["target_rel"],
            # observations["proprioception"]["contact"]
        ], dim=1)
        proprio_features = self.proprio_fc(proprio)
        
        # 特征融合
        return proprio_features
# 环境配置
env = DummyVecEnv([lambda: GraspEnv("./model/UR5+gripper/UR5gripper_2_finger.xml")])
env = VecFrameStack(env, n_stack=1)  # 帧堆叠

# 策略配置
policy_kwargs = {
    "features_extractor_class": CustomFeatureExtractor,
    "net_arch": dict(pi=[256,64], vf=[64,64]),
    "activation_fn": torch.nn.ReLU
}

# 训练参数
model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    verbose=1,
    tensorboard_log="./mujoco_src/logs/tensorboard/"
)

# 回调函数
checkpoint_cb = CheckpointCallback(
    save_freq=10000,
    save_path="./mujoco_src/logs/best_model"
)

# 开始训练
model.learn(
    total_timesteps=1_000_000,
    callback=checkpoint_cb,
    progress_bar=True
)