from collections import OrderedDict, defaultdict
import gymnasium as gym
import mujoco
import numpy as np
from termcolor import colored
from gymnasium import spaces
from camera import Camera
from robot import Robot
from mujoco import viewer
import traceback
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200


class GraspEnv(gym.Env):
    def __init__(self, model_path,max_episode_steps = 500):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = viewer.launch_passive(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        for _ in range(1000):
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
        
        self.robot = Robot(model=self.model, data=self.data, viewer=self.viewer)
        self.camera = Camera(self.model, self.data, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, camera_name="top_down")
        self.real_obj_low = np.array([-0.1, -0.65])
        self.real_obj_high = np.array([0.1, -0.45])
        self.real_action_low = np.array([-0.15, -0.7,1.1])
        self.real_action_high = np.array([0.15, -0.4,1.1])
        self.observation_space = spaces.Dict(OrderedDict([
            ("obj_pos", spaces.Box(low=self.real_obj_low, high=self.real_obj_high, shape=(2,))),
            ("eef_pos", spaces.Box(low=self.real_action_low[0:2], high=self.real_action_high[0:2], shape=(2,))),
            # ("gripper", spaces.Box(low=-0.5, high=0.5, shape=(1,))),
            # ("target_rel", spaces.Box(low=0, high=2, shape=(3,))),
            # ("contact", spaces.Box(low=0, high=10, shape=(6,)))
        ]))
        self.action_space = spaces.Discrete(4)
        self.current_step = 0
        self.max_episode_steps=max_episode_steps

          

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_1")
        joint_adr = self.model.jnt_qposadr[self.model.body_jntadr[object_id]]
        x = np.random.uniform(self.real_obj_low[0], self.real_obj_high[0])
        y = np.random.uniform(self.real_obj_low[1], self.real_obj_high[1])
        z = 0.95

        body_offset = self.model.body_pos[object_id]
        local_qpos = np.array([x, y, z]) - body_offset

        self.data.qpos[joint_adr:joint_adr+3] = local_qpos
        self.data.qpos[joint_adr+3:joint_adr+7] = [1, 0, 0, 0]
        self.data.qvel[self.model.jnt_dofadr[self.model.body_jntadr[object_id]]:
                    self.model.jnt_dofadr[self.model.body_jntadr[object_id]]+6] = 0

        print(f"🎯 Object world position target: ({x:.3f}, {y:.3f}, {z:.3f})")

        for _ in range(1000):
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

        self.robot.reset()
        self.current_step = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        success = self._apply_action(action)
        obs = self._get_obs()
        reward, _ = self._compute_reward(obs, action)
        terminated = success
        truncated = (self.current_step >= self.max_episode_steps)
        self.current_step += 1
        info = {"is_success": success}
        return obs, reward, terminated, truncated, info

            
    
    
    def _apply_action(self, action):
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_1")
        eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        ee_pos = self.data.xpos[eef_id].copy()
        obj_pos = self.data.xpos[object_id].copy()
        horizontal_dist = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
        if horizontal_dist>=0.05:delta = 0.05
        else: delta=0.02
        action_map = {
            0: np.array([-delta,  0.0,   0.0]),  # left
            1: np.array([ 0.0,   -delta, 0.0]),
            2: np.array([ delta,  0.0,   0.0]),  # right
            3: np.array([ 0.0,    delta, 0.0]),  # backward
        }
        action = action_map[int(action)]
        new_pos = ee_pos + action
        clipped_pos = np.clip(new_pos, self.real_action_low, self.real_action_high)

        self.robot.move_ee(clipped_pos, pid=False)

        # 判断是否抓取
        ee_pos = self.data.xpos[eef_id].copy()
        obj_pos = self.data.xpos[object_id].copy()
        horizontal_dist = np.linalg.norm(ee_pos[:2] - obj_pos[:2])

        if horizontal_dist <= 0.02:
            self.robot.open_gripper()
            self.robot.stay(100)
            ee_pos = self.data.xpos[eef_id].copy()
            self.robot.move_ee([ee_pos[0], ee_pos[1], 0.932])
            self.robot.stay(100)
            self.robot.close_gripper()
            self.robot.stay(100)
            self.robot.move_ee([0.0, -0.6, 1.1])
            self.robot.stay(100)
            return True
        return False
 
        
    def _get_obs(self):
        # rgb, depth = self.camera.get_image_data(show=True)
        eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_1")
        obj_pos=self.data.xpos[object_id].copy()[0:2]
        eef_pos=self.data.xpos[eef_id].copy()[0:2]
        proprio = OrderedDict({
            "obj_pos":obj_pos,
            "eef_pos": eef_pos,
            # "gripper": self.data.qpos[6].copy(),
            # "target_rel": self._get_target_relative_pos(),
            # "contact": self.data.sensor_data[:6].copy()
        })
        return proprio

    def _compute_reward(self, obs, action):
        success = False
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_1")
        eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        obj_pos = self.data.xpos[object_id].copy()
        ee_pos = self.data.xpos[eef_id].copy()

        # --- 距离奖励 ---
       # 计算距离
        rel_pos = ee_pos-[0,0, 0.16] - obj_pos
        xy_dist = np.linalg.norm(rel_pos[:2])
        x_dist=abs(rel_pos[0])
        y_dist=abs(rel_pos[1])
        # 渐进式距离奖励
        dist_reward = 0.25-10*xy_dist -3*x_dist-3*y_dist

        # --- 动作惩罚 ---
        # action_penalty = 0.01 * np.sum(np.square(action))

        # --- 抓取奖励 ---
        gripper_pos = self.data.qpos[6]
        gripper_closed = gripper_pos < -0.2  # 夹爪闭合阈值
        grasp_reward=0
        if gripper_closed:
            grasp_reward = 0.5  
            print("grasp_reward")

        # --- 成功判断 ---
        object_lifted = obj_pos[2] > 1 # 初始高度约 1.0
        if gripper_closed and object_lifted:
            success = True
            print("success_reward")
            grasp_reward += 0.5  # 抬起物体额外奖励

        reward =  dist_reward + grasp_reward 

        return reward, success
