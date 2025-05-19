from collections import OrderedDict, defaultdict
import gym
import mujoco
import numpy as np
from termcolor import colored
from gym import  spaces
from camera import Camera
from robot import Robot
from mujoco import viewer
IMAGE_WIDTH=200
IMAGE_HEIGHT=200


class GraspEnv(gym.Env):
    def __init__(self, model_path):
        # 加载模型和数据
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer=viewer.launch_passive(self.model, self.data)
        
        self.robot = Robot(model=self.model,data=self.data,viewer=self.viewer)
        #self.camera= Camera(self.model,self.data,width=IMAGE_WIDTH,height=IMAGE_HEIGHT, camera_name="top_down")
        
        self.observation_space = spaces.Dict(OrderedDict([
        #    ("rgb",spaces.Box(low=0, high=255, shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)),
        #    ( "depth",spaces.Box(low=0.0, high=5.0, shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)),
         
        ("joint_pos",spaces.Box(low=-np.pi,high=np.pi,shape=(6,))),
        ("eef_pos", spaces.Box(low=-2, high=2, shape=(3,))),
        # ("gripper", spaces.Box(low=0, high=1, shape=(1,))),
        ("target_rel", spaces.Box(low=-2, high=2, shape=(3,))),
        # ("contact", spaces.Box(low=0, high=10, shape=(6,)))
        
        ]))
        # self.action_space = spaces.Dict(OrderedDict([
            # ("motion",spaces.Box(low=-1,high=1,shape=(6,),dtype=np.float32)),
        # ]))
        self.action_space =spaces.Box(low=-1,high=1,shape=(6,),dtype=np.float32)
        #训练参数
        self.current_step=0
        self.max_episode_steps=3000
        self.reset()
        

    def reset(self):
        self.robot.reset()
        mujoco.mj_resetData(self.model, self.data)
        self.current_step=0
        return self._get_obs()
    
    def _apply_action(self,action):
        #self.robot.move_group_to_joint_target_in_one_mjstep(group="Arm",target=action)
        self.robot._move_group_to_joint_target(group="Arm",target=action)
    
    
    def step(self, action):
        self._apply_action(action)
        obs = self._get_obs()
        reward=0
        success=False
        # reward,success=self._compute_reward(obs,action)
        #done = success or (self.current_step >= self.max_episode_steps)
        done=success
        self.current_step+=1
        
        return obs,reward,done,{"is_success": success}
    
    def _get_target_relative_pos(self):
        # 计算目标相对位置
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_1")
        eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        
        return self.data.xpos[eef_id].copy()-self.data.xpos[object_id].copy()
        

    def _get_obs(self):
        #rgb, depth = self.camera.get_image_data(show=True)
        eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        proprio = OrderedDict({
            "joint_pos": self.data.qpos[:6].copy(),
            "eef_pos": self.data.xpos[eef_id].copy(),
            # "gripper": self.data.qpos[6].copy(),
            "target_rel": self._get_target_relative_pos(),
            #"contact": self.data.sensor_data[:6].copy()
            
         })
        return proprio

    def _compute_reward(self,obs,action):
        dist = np.linalg.norm(obs["target_rel"])
        dist_reward = 1.0 / (1.0 + 10.0 * dist**2)
        # 接触奖励
        contact_reward = 0
        #0.5 if np.any(obs["proprioception"]["contact"] > 0.1) else 0
        # 抓取成功判断
        success = False
        success_reward = 0
        # success = self._check_grasp_success(obs)
        # success_reward = 10.0 if success else 0

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
            # obs["proprioception"]["gripper"] < 0.05   # 夹爪闭合
            # and 
            np.linalg.norm(obs["target_rel"]) < 0.03  # 目标接近
            and obs["eef_pos"][2] > 0.15  # 已提起
            #and  np.mean(obs["proprioception"]["contact"]) > 0.5  # 持续接触
        )