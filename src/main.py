
import numpy as np
import pybullet as p
import os

import pybullet_data
from data import YCBModels
from env import GraspTaskEnv
from robot import ArmBase
from task2 import CustomCNN, UR5GraspingEnv
from util import Camera
from task import task_step
import numpy as np
import pybullet as p
import os
import gym
from gym import spaces
import pybullet_data
import time
from collections import deque, namedtuple
from attrdict import AttrDict
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
def demo1():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    ycb_models = YCBModels(
        os.path.join('/home/wy/Document/eibeginner/model/pybullet_ur5_robotiq/data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = Camera((0, -0.5, 1.5), 0.1, 5, (320, 320), 40)
    robot = ArmBase("/home/wy/Document/eibeginner/model/pybullet_ur5_robotiq/urdf/ur5_robotiq_85.urdf")
    env= GraspTaskEnv(ycb_models, camera,robot,obj_num=2, vis=True)
    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
    (rgb, depth, seg) = env.reset()
    step_cnt = 0
    while True:
        h_, w_ = np.unravel_index(depth.argmin(), depth.shape)
        x, y, z = camera.rgbd_2_world(w_, h_, depth[h_, w_])

        p.addUserDebugLine([x, y, 0], [x, y, z], [0, 1, 0])
        p.addUserDebugLine([x, y, z], [x, y, z+0.05], [1, 0, 0])

        (rgb, depth, seg), reward, done, info = task_step(env,(x, y, z), 1)

        print('Step %d, grasp at %.2f,%.2f,%.2f, reward %f, done %s, info %s' %
              (step_cnt, x, y, z, reward, done, info))
        step_cnt += 1

# 训练函数
def train():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    # 创建环境
    ycb_models = YCBModels(
        os.path.join('/home/wy/Document/eibeginner/model/pybullet_ur5_robotiq/data/ycb', '**', 'textured-decmp.obj')
    )
    camera = Camera((0, -0.5, 1.5), 0.1, 5, (320, 320), 40)
    robot = ArmBase("/home/wy/Document/eibeginner/model/pybullet_ur5_robotiq/urdf/ur5_robotiq_85.urdf")
    
    env = UR5GraspingEnv(ycb_models, camera, robot, obj_num=1, vis=True)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # 策略网络配置
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,  # 可考虑线性衰减
        n_steps=2048,        # 每次更新前收集的步数
        batch_size=256,       # 经验回放缓冲区大小
        n_epochs=5,          # 每次更新的epoch数
        gamma=0.99,          # 折扣因子(长期回报)
        gae_lambda=0.95,     # GAE参数(权衡偏差方差)
        clip_range=0.2,      # 策略更新裁剪范围
        ent_coef=0.01,       # 熵系数(鼓励探索)
        max_grad_norm=0.5,   # 梯度裁剪阈值
        tensorboard_log="./logs/"  # 添加TensorBoard日志
    )

    # 增强版评估回调
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10000,      # 每10000步评估一次
        deterministic=True,
        render=False,
        n_eval_episodes=5    # 每次评估5个episode
    )

    # 训练配置
    total_timesteps = int(1e6)  # 初始100万步
    model.learn(total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="ur5_grasping",# TensorBoard实验名称
        progress_bar=True)
        
    # 保存模型
    model.save("ur5_grasping_ppo")
    env.save("ur5_grasping_env.pkl")

# 演示函数
def demo2(model_path="ur5_grasping_ppo"):
    # 创建环境
    ycb_models = YCBModels(
        os.path.join('/home/wy/Document/eibeginner/model/pybullet_ur5_robotiq/data/ycb', '**', 'textured-decmp.obj')
    )
    camera = Camera((0, -0.5, 1.5), 0.1, 5, (320, 320), 40)
    robot = ArmBase("/home/wy/Document/eibeginner/model/pybullet_ur5_robotiq/urdf/ur5_robotiq_85.urdf")
    
    env = UR5GraspingEnv(ycb_models, camera, robot, obj_num=2, vis=True)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load("ur5_grasping_env.pkl", env)
    
    # 加载模型
    model = PPO.load(model_path, env=env)
    
    # 运行演示
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        # 可视化
        if dones:
            obs = env.reset()   

if __name__ == '__main__':
   train()