import numpy as np
import mujoco
from gym import spaces
import cv2
from collections import OrderedDict

class UR5GraspEnv(gym.Env):
    def __init__(self, model_path, image_size=(84, 84)):
        # 初始化MuJoCo
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        # 观测空间配置
        self.image_size = image_size
        self.observation_space = spaces.Dict(OrderedDict([
            ("rgb", spaces.Box(low=0, high=1, shape=(3, *image_size), dtype=np.float32)),
            ("depth", spaces.Box(low=0, high=1, shape=image_size, dtype=np.float32)),
            ("proprioception", spaces.Dict(OrderedDict([
                ("joint_pos", spaces.Box(low=-np.pi, high=np.pi, shape=(6,))),
                ("joint_vel", spaces.Box(low=-5, high=5, shape=(6,))),
                ("eef_pos", spaces.Box(low=-2, high=2, shape=(3,))),
                ("eef_quat", spaces.Box(low=-1, high=1, shape=(4,))),
                ("gripper", spaces.Box(low=0, high=1, shape=(1,))),
                ("target_rel", spaces.Box(low=-2, high=2, shape=(3,))),
                ("contact", spaces.Box(low=0, high=10, shape=(6,)))
            ])))
        ]))
        
        # 动作空间配置
        self.action_space = spaces.Dict({
            "motion": spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
            "gripper": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        
        # 训练参数
        self.max_episode_steps = 200
        self.current_step = 0
        
        # 初始化状态
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        return self._get_obs()
    
    def step(self, action):
        # 动作处理
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)
        
        # 获取观测
        obs = self._get_obs()
        
        # 计算奖励
        reward, success = self._compute_reward(obs, action)
        
        # 终止条件
        done = success or (self.current_step >= self.max_episode_steps)
        self.current_step += 1
        
        return obs, reward, done, {"is_success": success}

    def _get_obs(self):
        # 视觉观测处理
        rgb, depth = self._get_camera_data()
        rgb = cv2.resize(rgb, self.image_size).transpose(2, 0, 1) / 255.0
        depth = cv2.resize(depth, self.image_size) / 10.0
        
        # 本体感知处理
        proprio = {
            "joint_pos": self.data.qpos[:6].copy(),
            "joint_vel": self.data.qvel[:6].copy(),
            "eef_pos": self.data.site_xpos[self.model.site_name2id("tool_center")].copy(),
            "eef_quat": self.data.site_xquat[self.model.site_name2id("tool_center")].copy(),
            "gripper": np.array([self.data.actuator_length[6]]),
            "target_rel": self._get_target_relative_pos(),
            "contact": self.data.sensor_data[:6].copy()
        }
        
        return OrderedDict([
            ("rgb", rgb.astype(np.float32)),
            ("depth", depth.astype(np.float32)),
            ("proprioception", proprio)
        ])

    def _apply_action(self, action):
        # 笛卡尔空间控制
        delta_pos = action["motion"][:3] * 0.05  # 最大5cm位移
        delta_rot = action["motion"][3:] * 0.2   # 最大0.2弧度旋转
        
        # 计算新位姿
        current_pos = self.data.site_xpos[self.model.site_name2id("tool_center")].copy()
        target_pos = current_pos + delta_pos
        
        # 使用MuJoCo内置IK
        self.model.site_pos[self.model.site_name2id("target_site")] = target_pos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_inverse(self.model, self.data)
        
        # 应用关节控制
        self.data.ctrl[:6] = self.data.qpos[:6] + self.data.qvel[:6] * 0.1  # 加入速度项
        
        # 夹爪控制
        self.data.ctrl[6] = action["gripper"][0] * 0.1  # 映射到实际范围

    def _compute_reward(self, obs, action):
        # 核心奖励组件
        dist = np.linalg.norm(obs["proprioception"]["target_rel"])
        dist_reward = 1.0 / (1.0 + 10.0 * dist**2)
        
        # 接触奖励
        contact_reward = 0.5 if np.any(obs["proprioception"]["contact"] > 0.1) else 0
        
        # 抓取成功判断
        success = self._check_grasp_success(obs)
        success_reward = 10.0 if success else 0
        
        # 动作平滑惩罚
        action_penalty = 0.01 * np.sum(np.square(action["motion"]))
        
        # 总奖励
        total_reward = (
            0.3 * dist_reward +
            0.2 * contact_reward +
            success_reward -
            action_penalty
        )
        
        return total_reward, success

    def _check_grasp_success(self, obs):
        return (
            obs["proprioception"]["gripper"][0] < 0.05 and  # 夹爪闭合
            np.linalg.norm(obs["proprioception"]["target_rel"]) < 0.03 and  # 目标接近
            obs["proprioception"]["eef_pos"][2] > 0.15 and  # 已提起
            np.mean(obs["proprioception"]["contact"]) > 0.5  # 持续接触
        )

    def _get_camera_data(self):
        # 实现获取RGB和深度图像
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3)), np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

    def _get_target_relative_pos(self):
        # 计算目标相对位置
        return np.zeros(3)


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

# 环境配置
env = DummyVecEnv([lambda: UR5GraspEnv("ur5_grasp.xml")])
env = VecFrameStack(env, n_stack=3)  # 帧堆叠

# 策略配置
policy_kwargs = {
    "features_extractor_class": CustomFeatureExtractor,
    "net_arch": [dict(pi=[256,256], vf=[256,256])],
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
    tensorboard_log="./grasp_tensorboard/"
)

# 回调函数
checkpoint_cb = CheckpointCallback(
    save_freq=10000,
    save_path="./grasp_models/"
)

# 开始训练
model.learn(
    total_timesteps=1_000_000,
    callback=checkpoint_cb,
    progress_bar=True
)