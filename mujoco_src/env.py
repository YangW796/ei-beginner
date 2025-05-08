from collections import defaultdict
import mujoco
import mujoco.viewer
import numpy as np
from termcolor import colored
from gym import  spaces
from robot import Robot
from mujoco import viewer
IMAGE_WIDTH=200
IMAGE_HEIGHT=200


class GraspEnv:
    def __init__(self, model_path):
        # 加载模型和数据
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        #viewer.launch(self.model,self.data)

        self.robot = Robot(model=self.model,data=self.data)
        #self.robot.display_current_values()
        
        
        self.current_observation={
        "rgb":np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
        "depth":np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT))
        }
        self.step_called=0
        


    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()
    
    def _is_done(self):
        return False
    

    def step(self, action):
        self.current_observation = self._get_obs()
        # 计算位置
        x = action[0] % IMAGE_WIDTH
        y = action[0] // IMAGE_WIDTH
        rotation = action[1]
        depth = self.current_observation["depth"][y][x]
        coordinates = self.controller.pixel_2_world(
                pixel_x=x, pixel_y=y, depth=depth, height=IMAGE_HEIGHT, width=IMAGE_WIDTH
        )
        
        #  移动+是否完成
        done=self.robot.move_and_grasp(coordinates)
         
        # 计算奖励
        reward=self._compute_reward(coordinates,rotation)
        # 获取当前obs
        self.current_observation = self._get_obs()
        self.step_called+=1
        
        
        

        return self.current_observation, reward, done, {}
    
    def _set_action_space(self):
        if self.action_space_type == "discrete":
            size = IMAGE_WIDTH * IMAGE_HEIGHT
            self.action_space = spaces.Discrete(size)
        elif self.action_space_type == "multidiscrete":
            self.action_space = spaces.MultiDiscrete(
                [IMAGE_HEIGHT * IMAGE_WIDTH, len(self.rotations)]
            )

        return self.action_space

    def _get_obs(self):
        # rgb, depth = self.controller.get_image_data(
        #     width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, show=show
        # )
        depth = self.controller.depth_2_meters(depth)
        observation = defaultdict()
        observation["rgb"] = rgb
        observation["depth"] = depth

        return observation

    def _compute_reward(self, coordinates,rotation):
        if coordinates[2] < 0.8 or coordinates[1] > -0.3:
            reward=0
        else:
            reward = 1 #if grasped_something else 0
            # if grasped_something != "demo":
            #     print(
            #         colored(
            #             "Reward received during step: {}".format(reward),
            #             color="yellow",
            #             attrs=["bold"],
            #         )
            #     )
    
        return reward  # 比如奖励为 X 方向移动量


    def render(self):
        mujoco.viewer.launch_passive(self.model, self.data)