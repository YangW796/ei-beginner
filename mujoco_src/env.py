import mujoco
import mujoco.viewer
import numpy as np


class MujocoEnvDemo:
    def __init__(self, model_path):
        # 加载模型和数据
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = False  # 可设置 done 条件
        return obs, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _compute_reward(self, obs):
        # 示例奖励函数
        return obs[0]  # 比如奖励为 X 方向移动量

    def render(self):
        mujoco.viewer.launch_passive(self.model, self.data)