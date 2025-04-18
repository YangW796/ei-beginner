
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
GRIPPER_MOVING_HEIGHT = 1.15
SIMULATION_STEP_DELAY = 1 / 240.
GRASP_POINT_OFFSET_Z = 1.231 - 1.1
GRASP_SUCCESS_REWARD = 1
GRASP_FAIL_REWARD = -0.3
GRIPPER_GRASPED_LIFT_HEIGHT = 1.2
# 强化学习环境类
class UR5GraspingEnv(gym.Env):
    def __init__(self, models, camera, robot, obj_num=3, vis=False):
        super(UR5GraspingEnv, self).__init__()
        self.step_count=0
        # 初始化参数
        self.models = models
        self.camera = camera
        self.robot = robot
        self.obj_num = obj_num
        self.vis = vis

        # 动作空间: [dx, dy, dz, gripper_action]
        self.action_space = spaces.Box(
            low=np.array([-0.224, -0.724, 1, 0]),  # 相对位移和夹爪动作
            high=np.array([0.224, -0.276, 1.3, 0.085]),
            dtype=np.float32
        )
        
        # 观测空间: 深度图像 + 机器人状态
        self.observation_space = spaces.Dict({
            'depth': spaces.Box(low=0, high=1, shape=camera.resolution, dtype=np.float32),
            'joint_positions': spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32),
            'gripper_state': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        
        # 初始化物理环境
        self._init_physics()
        
        # 加载物体
        self.objects = self._load_objects()
        self.robot.reset_robot()
        
        # 可视化设置
        if self.vis:
            p.resetDebugVisualizerCamera(2.0, -270, -60, (0, 0, 0))
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    
    def _init_physics(self):
        
        # 加载平面和桌子
        self.planeID = p.loadURDF("plane.urdf")
        self.tableID = p.loadURDF(
            "/home/wy/Document/eibeginner/model/pybullet_ur5_robotiq/urdf/objects/table.urdf",
            [0.0, -0.5, 0.8],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True
        )
        self.UR5StandID = p.loadURDF(
            "/home/wy/Document/eibeginner/model/pybullet_ur5_robotiq/urdf/objects/ur5_stand.urdf",
            [-0.7, -0.36, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True
        )
    
    def _load_objects(self):
        return self.models.load_objects(self.obj_num)
        
    def _get_observation(self):
        # 获取深度图像
        _, depth, _ = self.camera.shot()
        depth = np.array(depth, dtype=np.float32)
        depth = (depth - self.camera.near) / (self.camera.far - self.camera.near)
        
        # 获取机器人状态
        joint_states = p.getJointStates(self.robot.robot_id, range(6))  # 前6个关节
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)
        
        # 获取夹爪状态
        gripper_state = np.array([p.getJointState(self.robot.robot_id, self.robot.joints['finger_joint'].id)[0]], dtype=np.float32)
        
        return {
            'depth': depth,
            'joint_positions': joint_positions,
            'gripper_state': gripper_state
        }
    
    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        self._init_physics()
        # 加载物体
        self.objects = self._load_objects()
        self.robot.reset_robot()
        p.resetDebugVisualizerCamera(2.0, -270, -60, (0, 0, 0))
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        return self._get_observation()
    
    def step(self, action):
        
        dx, dy, dz, gripper_action = action
        
        current_pos, current_orn = p.getLinkState(self.robot.robot_id, self.robot.eefID)[:2]
        target_pos = [current_pos[0] + dx, current_pos[1] + dy,GRIPPER_MOVING_HEIGHT]
        

        success,(final_xyz,_) = self.robot.move_ee([target_pos[0], target_pos[1], GRIPPER_MOVING_HEIGHT, current_orn],custom_velocity=0.05)
        self.robot.open_gripper()
        self.robot.move_ee((target_pos[0],  target_pos[1], GRASP_POINT_OFFSET_Z, current_orn), try_close_gripper=False,custom_velocity=0.05)
        item_in_gripper =self.robot.close_gripper()
        self.robot.move_ee((target_pos[0],  target_pos[1],GRASP_POINT_OFFSET_Z + 0.1, current_orn), try_close_gripper=False,custom_velocity=0.05)
        self.robot.move_away_arm()
        self.robot.open_gripper()
        # self.camera.position=final_xyz
        # 获取新状态
        obs = self._get_observation()
        
        # 计算奖励
        reward = self._compute_reward(final_xyz,target_pos,current_pos)
        
        # 检查是否完成
        done = self._check_success()
        
        # 额外信息
        info = {'success': self._check_success()}
        
        return obs, reward, done, info
    
    def _compute_reward(self,final_xyz,target_pos,current_pos):
        reward = 0
        dist_to_target = np.linalg.norm(np.array(final_xyz) - np.array(target_pos))
        reward += 0.1 - dist_to_target # 距离越近，奖励越高
        
        # 1. 检查是否接触到物体
        contacts = p.getContactPoints(bodyA=self.robot.robot_id)
        for contact in contacts:
            if contact[2] != self.planeID and contact[2] != self.tableID:
                reward += 2  # 接触奖励
        
        # 2. 检查是否成功抓取物体
        for obj_id in self.objects:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            if pos[2] > 0.2:  # 物体被举起来了
                reward += 4  # 成功奖励
        
        
        # 3. 惩罚大动作
       # reward -= 0.01 * np.sum(np.square(self.last_action[:3])) if hasattr(self, 'last_action') else 0
        
        return reward
    
    def _check_success(self):
        for obj_id in self.objects:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            if pos[2] > 0.2:  # 物体被举起来了
                return True
        return False
    
    def close(self):
        p.disconnect()

# 自定义特征提取器
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)
        
        # 假设输入深度图像为320x320
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # 输出: (320-8)/4+1 = 79
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 输出: (79-4)/2+1 = 38
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 输出: (38-3)/1+1 = 36
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 计算CNN输出维度
        with torch.no_grad():
            sample = torch.as_tensor(observation_space['depth'].sample()[None, None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        # 机器人状态处理
        robot_state_dim = observation_space['joint_positions'].shape[0] + \
                         observation_space['gripper_state'].shape[0]
        
        # 合并特征的全连接层
        self.robot_state_net = nn.Sequential(
            nn.Linear(robot_state_dim, 64),
            nn.ReLU()
        )
        
        # 最终全连接层
        self.fc = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 处理深度图像
        depth_tensor = observations['depth']
        if depth_tensor.dim() == 3:  # 添加通道维度
            depth_tensor = depth_tensor.unsqueeze(1)
        depth_features = self.cnn(depth_tensor)
        
        # 处理机器人状态
        robot_state = torch.cat([
            observations['joint_positions'],
            observations['gripper_state']
        ], dim=-1)
        robot_features = self.robot_state_net(robot_state)
        
        # 合并特征
        combined_features = torch.cat([depth_features, robot_features], dim=-1)
        return self.fc(combined_features)