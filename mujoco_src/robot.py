from collections import defaultdict
import copy
import time
import mujoco
import numpy as np
from simple_pid import PID
from ikpy.chain import Chain
from math import cos, sin, atan2, acos, sqrt
TABLE_HEIGHT=0.87
class Robot:
    def __init__(self,path=None,model=None,data=None,viewer=None):
        self.model= model
        self.data = data
        self.viewer=viewer
        self._init_pid_list()
        # 组序号
        self.groups = defaultdict(list)
        self.groups["Arm"] = list(range(5))
        self.groups["Gripper"]=[6]
        self.actuated_joint_ids = np.array([i[2] for i in self.actuators]).astype(int)
        self.ee_chain = Chain.from_urdf_file("./model/UR5+gripper/ur5_gripper.urdf")
        self.reset()
        
    
    def reset(self):
        self.data.qpos[:7] = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0, 0.4])
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()
    
    def _init_pid_list(self):
        self.pid_list = []
        sample_time = 0.0001
        p_scale = 3
        i_scale = 0.0
        i_gripper = 0
        d_scale = 0.1

        # 定义每个关节的 PID 参数
        pid_params = [
            (7 * p_scale, 0.0 * i_scale, 1.1 * d_scale, 0.0, (-2, 2)),     # Shoulder Pan: 0 rad
            (10 * p_scale, 0.0 * i_scale, 1.0 * d_scale, -np.pi/2, (-2, 2)), # Shoulder Lift: -π/2
            (5 * p_scale, 0.0 * i_scale, 0.5 * d_scale, np.pi/2, (-2, 2)),   # Elbow: π/2
            (7 * p_scale, 0.0 * i_scale, 0.1 * d_scale, -np.pi/2, (-1, 1)),  # Wrist1: -π/2
            (5 * p_scale, 0.0 * i_scale, 0.1 * d_scale, 0.0, (-1, 1)),       # Wrist2: 0
            (5 * p_scale, 0.0 * i_scale, 0.1 * d_scale, 0.0, (-1, 1)),       # Wrist3: 0
            (2.5 * p_scale, i_gripper, 0.0 * d_scale, 0.0, (-1, 1)),         # Gripper
        ]
        for kp, ki, kd, sp, out_lim in pid_params:
            self.pid_list.append(
                PID(kp, ki, kd, setpoint=sp, output_limits=out_lim, sample_time=sample_time)
            )
        
        # 设置初始目标 joint 值
        self.current_target_joint_values = np.array(
            [pid.setpoint for pid in self.pid_list]
        )
        
        # 构建 actuator pid映射信息
        self.actuators = []
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            joint_id = self.model.actuator_trnid[i][0]
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            controller = self.pid_list[i]
            self.actuators.append([i, actuator_name, joint_id, joint_name, controller])
    
    
    def open_gripper(self):
        return self._move_group_to_joint_target(
            group="Gripper", target=[0], max_steps=1000, tolerance=0.05
        )
    
    def close_gripper(self):
        return self._move_group_to_joint_target(
            group="Gripper", target=[-0.4], max_steps=1000, tolerance=0.05) 
        
    def _move_group_to_joint_target(
        self,
        group,
        target=None,
        tolerance=0.05,
        max_steps=3000
    ):
        idxs = self.groups[group]
        #初始化
        steps = 1
        reached_target = False
        deltas = np.zeros(self.model.nu)
        
        #设置 PID 控制器的 setpoint 
        for i, v in enumerate(idxs):
            self.current_target_joint_values[v] = target[i]
            
        for j in range(self.model.nu):
            self.actuators[j][4].setpoint =self.current_target_joint_values[j]
        
        #执行循环
        while not reached_target:
            current_joint_values = self.data.qpos[self.actuated_joint_ids]
            # 控制所有 motor
            for j in range(self.model.nu):
                self.data.ctrl[j] = self.actuators[j][4](current_joint_values[j])
            
            for i in idxs:
                deltas[i] = abs(self.current_target_joint_values[i] - current_joint_values[i])
            
            if steps % 1000 == 0:
                print(
                    "Moving group {} to joint target! Max. delta: {}, Joint: {}".format(
                        group, max(deltas), self.actuators[np.argmax(deltas)][3]
                    )
                )
            
            if max(deltas) < tolerance:
                # print(f"\033[1;32mJoint values for group {group} within requested tolerance! ({steps} steps)\033[0m")
                reached_target = True
                
            if steps > max_steps:
                # print(f"\033[1;31mMax number of steps reached: {max_steps}\033[0m")
                # print("Deltas: ", deltas)
                break
            
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()  
            # self.stay(1000)
            
            steps+=1   
        return reached_target,max(deltas)
    
    def mujoco_move_forward(self,target):
        current_pos=self.data.qpos[:6].copy()
        self.data.qpos[:6] =current_pos+target
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
    
    def _ik(self, ee_position):

        # Convert world target position to base frame
        base_pos = self.data.xpos[self.model.body('base_link').id]
        ee_position_base = ee_position - base_pos
        # Add offset to transform ee_link → gripper center
        gripper_center_position = ee_position_base + np.array([0, -0.005, 0.16])
    
        joint_angles = self.ee_chain.inverse_kinematics(
            gripper_center_position,
            target_orientation=[0,0,-1], 
            orientation_mode="X")
        # Check accuracy using forward kinematics
        fk_pos = self.ee_chain.forward_kinematics(joint_angles)[:3, 3]
        prediction = fk_pos + base_pos - np.array([0, -0.005, 0.16])
        error = np.linalg.norm(prediction - ee_position)

        joint_angles = joint_angles[1:-2]  # Remove fixed/base links

        if error < 0.02:
            return joint_angles
        else:
            return None



    def move_ee(self,ee_position):
        joint_angles = self._ik_2(ee_position) 
        return self._move_group_to_joint_target(group="Arm", target=joint_angles) if joint_angles is not None else None
    
    def stay(self,duration):
        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:
            elapsed = (time.time() - starting_time) * 1000
        
    
    def move_and_grasp(self, coordinates):

        coordinates_1 = copy.deepcopy(coordinates)
        coordinates_1[2] = 1.1
        result1,_= self.move_ee(coordinates_1)
        print("result1")
        self.stay(300)
        # result_rotate = self.rotate_wrist_3_joint_to_value(self.rotations[rotation])
        self.open_gripper()
        # Move to grasping height
        print("result2")
        self.stay(300)
        coordinates_2 = copy.deepcopy(coordinates)
        coordinates_2[2] = max(TABLE_HEIGHT, coordinates_2[2])
        print("coordinates_2",coordinates_2)
        self.move_ee(coordinates_2)
        print("result3")
        self.stay(300)
        
        result_grasp = self.close_gripper()
        print("result4")
        self.stay(300)
        # Move back above center of table
        self.move_ee([0.0, -0.6, 1.1])
        print("result5")
        self.stay(300)
        # Move to drop position
        self.move_ee([0.6, 0.0, 1.1])
        
        print("result6")
        self.stay(300)
        # Open gripper again
        self.open_gripper()
        print("result7")
        
        self.stay(300)
        return result_grasp
        
    def _ik_2(self, ee_position):
        """自定义逆动力学实现"""
        # UR5机械臂的DH参数
        a = [0, -0.425, -0.39225, 0, 0, 0]
        d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
        alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]
        
        # 转换为相对于基座的位置
        base_pos = self.data.xpos[self.model.body('base_link').id]
        x, y, z = ee_position - base_pos - np.array([0, -0.005, 0.16])  # 调整夹爪中心
        
        # 计算关节角度
        theta = np.zeros(6)
        
        # 关节1 (base rotation)
        theta[0] = atan2(y, x)
        
        # 关节3 (elbow)
        r = sqrt(x**2 + y**2)
        s = z - d[0]
        D = (r**2 + s**2 - a[1]**2 - a[2]**2) / (2 * a[1] * a[2])
        theta[2] = atan2(-sqrt(1 - D**2), D)
        
        # 关节2 (shoulder)
        theta[1] = atan2(s, r) - atan2(a[2]*sin(theta[2]), a[1] + a[2]*cos(theta[2]))
        
        # 计算手腕位置
        T01 = self._dh_matrix(theta[0], d[0], a[0], alpha[0])
        T12 = self._dh_matrix(theta[1], d[1], a[1], alpha[1])
        T23 = self._dh_matrix(theta[2], d[2], a[2], alpha[2])
        T03 = np.dot(np.dot(T01, T12), T23)
        
        # 手腕位置
        wrist_pos = T03[:3, 3]
        
        # 计算手腕方向
        wrist_orient = np.array([0, 0, -1])  # 简化假设
        
        # 关节4,5,6 (wrist)
        # 这里简化处理，实际应用中需要更精确的计算
        theta[3] = -np.pi/2  # 固定值
        theta[4] = 0          # 固定值
        theta[5] = 0          # 固定值
        
        # 返回前5个关节角度(忽略手腕旋转)
        return theta[:5]
    
    def _dh_matrix(self, theta, d, a, alpha):
        """计算DH变换矩阵"""
        ct = cos(theta)
        st = sin(theta)
        ca = cos(alpha)
        sa = sin(alpha)
        
        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])    
        
        
