from collections import OrderedDict, defaultdict
import gymnasium as gym
import mujoco
import numpy as np
from termcolor import colored
from gymnasium import spaces
from camera import Camera
from robot import Robot
from mujoco import viewer

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200


class GraspEnv(gym.Env):
    def __init__(self, model_path,max_episode_steps = 500):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = viewer.launch_passive(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        self.robot = Robot(model=self.model, data=self.data, viewer=self.viewer)
        # self.camera = Camera(self.model, self.data, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, camera_name="top_down")
        
        self.observation_space = spaces.Dict(OrderedDict([
            # ("rgb", spaces.Box(low=0, high=255, shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)),
            # ("depth", spaces.Box(low=0.0, high=5.0, shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)),
            ("joint_pos", spaces.Box(low=-np.pi, high=np.pi, shape=(6,))),
            ("eef_pos", spaces.Box(low=-2, high=2, shape=(3,))),
            # ("gripper", spaces.Box(low=0, high=1, shape=(1,))),
            ("target_rel", spaces.Box(low=-2, high=2, shape=(3,))),
            # ("contact", spaces.Box(low=0, high=10, shape=(6,)))
        ])) 
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        # 训练参数
        self.current_step = 0
        self.max_episode_steps = max_episode_steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.reset()
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        self._apply_action(action)
        obs = self._get_obs()
        reward, success = self._compute_reward(obs, action)
        terminated = success
        truncated = (self.current_step >= self.max_episode_steps)
        self.current_step += 1
        info = {"is_success": success}
        return obs, reward, terminated, truncated, info
    
    
    def _apply_action(self, action):
        # self.robot._move_group_to_joint_target(group="Arm", target=action)
        self.robot.move_group_to_joint_target_in_one_mjstep(group="Arm", target=action)
        
        # 位置判断：是否夹爪在物体上方，且距离足够近
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_1")
        eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")

        obj_pos = self.data.xpos[object_id]
        ee_pos = self.data.xpos[eef_id]

        horizontal_dist = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
        vertical_dist = ee_pos[2] - obj_pos[2]

        if horizontal_dist < 0.05 and 0 < vertical_dist < 0.1:
            # 自动闭合夹爪
            print("close_gripper")
            self.robot.close_gripper()
    

    
    def _get_target_relative_pos(self):
        # 计算目标相对位置
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_1")
        eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        
        return self.data.xpos[eef_id].copy() - self.data.xpos[object_id].copy()
        
    def _get_obs(self):
        # rgb, depth = self.camera.get_image_data(show=True)
        eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        proprio = OrderedDict({
            "joint_pos": self.data.qpos[:6].copy(),
            "eef_pos": self.data.xpos[eef_id].copy(),
            # "gripper": self.data.qpos[6].copy(),
            "target_rel": self._get_target_relative_pos(),
            # "contact": self.data.sensor_data[:6].copy()
        })
        return proprio

    def _compute_reward(self, obs, action):
        reward = 0

        # ===== 阶段 1：靠近目标 =====
        dist = np.linalg.norm(obs["target_rel"])
        reward += np.exp(-10 * dist)  # 距离越近，奖励越大（0~1）

        success=False
        action_penalty = 0.01 * np.sum(np.square(action))
        reward -= action_penalty

        return reward, success