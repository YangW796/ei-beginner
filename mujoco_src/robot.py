from collections import defaultdict
import copy
import math
import time
from ikpy.chain import Chain
import mujoco
import numpy as np
from termcolor import colored
from simple_pid import PID 
from mujoco import viewer
TABLE_HEIGHT = 0.91
class Robot:
    def __init__(self,path=None,model=None,data=None):
        self.model= model
        self.data = data
        self.viewer=viewer.launch_passive(self.model, self.data)
        #初始化pid控制列表
        self._init_pid_list()
        
        #定义手臂夹爪idx组
        self.groups = defaultdict(list)
        self.groups["Arm"] = list(range(5))
        self.groups["Gripper"]=[6]
        #实际joint_ids
        self.actuated_joint_ids = np.array([i[2] for i in self.actuators]).astype(int)
        self.ee_chain = Chain.from_urdf_file("./model/UR5+gripper/ur5_gripper.urdf")
        
        self._init_pos()
        #self.rotations = {0: 0, 1: 30, 2: 60, 3: 90, 4: -30, 5: -60}
        

    def _init_pid_list(self):
        """
        Initialize per-joint PID controllers and metadata using updated mujoco API.
        """
        self.pid_list = []
        # PID parameters
        sample_time = 0.0001
        p_scale = 3
        i_scale = 0.0
        i_gripper = 0
        d_scale = 0.1

        # 定义每个关节的 PID 参数（你可以根据实际 URDF 配置来调整数量）
               # 定义每个关节的 PID 参数和目标角度（自然姿态）
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
        
    def _init_pos(self):
        initial_joint_angles = [0,-np.pi/2,np.pi/2,-np.pi/2,-np.pi/2,0,0]
        move_seq=[6,5,4,1,2,3,0]
        for j in range(self.model.nu):
            self.actuators[j][4].setpoint =initial_joint_angles[j]
        reached_target = False
        tolerance=0.01
        deltas = np.zeros(self.model.nu)
        
        while not reached_target:
            current_joint_values = self.data.qpos[self.actuated_joint_ids]
            for j in move_seq:
                self.data.ctrl[j] = self.actuators[j][4](current_joint_values[j])
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync() 
                
            
            for i in move_seq:
                deltas[i] = abs(initial_joint_angles[i]- current_joint_values[i])
            
            if max(deltas) < tolerance:
                reached_target = True
        
        print("init_pos finished")

    def _move_group_to_joint_target(
        self,
        group,
        target=None,
        tolerance=0.1,
        max_steps=10000
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
                print(f"\033[1;32mJoint values for group {group} within requested tolerance! ({steps} steps)\033[0m")
                reached_target = True
                
            if steps > max_steps:
                print(f"\033[1;31mMax number of steps reached: {max_steps}\033[0m")
                print("Deltas: ", deltas)
                break
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()  
            
            steps+=1
            
        return reached_target
            
    def open_gripper(self):
        return self._move_group_to_joint_target(
            group="Gripper", target=[0.4], max_steps=1000, tolerance=0.05
        )
    
    def close_gripper(self):
        return self._move_group_to_joint_target(
            group="Gripper", target=[-0.4], tolerance=0.01) 
    


    def _ik(self, ee_position):

        # Save current target
        self.current_carthesian_target = ee_position.copy()

        # Convert world target position to base frame
        base_pos = self.data.xpos[self.model.body('base_link').id]
        ee_position_base = ee_position - base_pos
        print(f"ee_pos:{ ee_position_base}")
        # Add offset to transform ee_link → gripper center
        gripper_center_position = ee_position_base + np.array([0, -0.005, 0.16])
        print(f"gripper_center_position:{ gripper_center_position}")
        # Solve IK using ikpy
        print(gripper_center_position)
        joint_angles = self.ee_chain.inverse_kinematics(
            gripper_center_position,
            [0, 0, -1],
            #orientation_mode="X",
        )

        # Check accuracy using forward kinematics
        fk_pos = self.ee_chain.forward_kinematics(joint_angles)[:3, 3]
        prediction = fk_pos + base_pos - np.array([0, -0.005, 0.16])
        error = np.linalg.norm(prediction - ee_position)

        joint_angles = joint_angles[1:-2]  # Remove fixed/base links

        if error < 0.02:
            return joint_angles
        else:
            print("IK error too high:", error)
            return None
    
    
    def move_ee(self,ee_position):
        joint_angles = self._ik(ee_position)
        return self._move_group_to_joint_target(group="Arm", target=joint_angles)
        
        
        
        
        
    
    def rotate_wrist_3_joint_to_value(self, degrees):
        pass
        
    def stay(self,duration):
        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:
            #self.move_group_to_joint_target(
            #     max_steps=10, tolerance=0.0000001, plot=False, quiet=True
            # )
            elapsed = (time.time() - starting_time) * 1000
        
    
    def move_and_grasp(self, coordinates):

        # Try to move directly above target
        coordinates_1 = copy.deepcopy(coordinates)
        coordinates_1[2] = 1.1
        result1 = self.move_ee(coordinates_1)
        # result_rotate = self.rotate_wrist_3_joint_to_value(self.rotations[rotation])

        self.open_gripper()
        # Move to grasping height
        coordinates_2 = copy.deepcopy(coordinates)
        coordinates_2[2] = max(TABLE_HEIGHT, coordinates_2[2] - 0.01)
        result2 = self.move_ee(coordinates_2)
        
        self.stay(100)
        result_grasp = self.close_gripper()

        # Move back above center of table
        _result3 = self.move_ee([0.0, -0.6, 1.1])

        # Move to drop position
        _result4 = self.move_ee([0.6, 0.0, 1.15])
        self.stay(500)

        result_final = self.close_gripper()
        # Open gripper again
        _result_open_again = self.open_gripper()
        self.stay(200)
        return result_final and result_grasp
        
        # Move back to zero rotation
        # result_rotate_back = self.rotate_wrist_3_joint_to_value(0)
        

    def display_current_values(self):
        """
        Debug method: Displays relevant simulation data using the latest MuJoCo API.
        Assumes self.model: mujoco.MjModel and self.data: mujoco.MjData
        """
        print("\n###############################")
        print("CURRENT JOINT POSITIONS (ACTUATED)")
        print("###############################")
        for i, joint_id in enumerate(self.actuated_joint_ids):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            qpos_idx = self.model.jnt_qposadr[joint_id]
            print(f"Joint {joint_name} (ID {joint_id}): angle = {self.data.qpos[qpos_idx]}")

        print("\n###############################")
        print("CURRENT JOINT POSITIONS (ALL)")
        print("###############################")
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            qpos_addr = self.model.jnt_qposadr[joint_id]
            size = self.model.jnt_dofadr[joint_id+1] - self.model.jnt_dofadr[joint_id] if joint_id + 1 < self.model.njnt else len(self.data.qpos) - qpos_addr
            val = self.data.qpos[qpos_addr:qpos_addr+size]
            print(f"Joint {name}: qpos = {val}")

        print("\n###############################")
        print("CURRENT BODY POSITIONS")
        print("###############################")
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            pos = self.data.xpos[body_id]
            print(f"Body {name}: position = {pos}")

        print("\n###############################")
        print("CURRENT BODY ROTATION MATRICES")
        print("###############################")
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            mat = self.data.xmat[body_id].reshape((3, 3))
            print(f"Body {name}: rotation matrix =\n{mat}")

        print("\n###############################")
        print("CURRENT BODY ROTATION QUATERNIONS (w, x, y, z)")
        print("###############################")
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            quat = self.data.xquat[body_id]
            print(f"Body {name}: quaternion = {quat}")

        print("\n###############################")
        print("CURRENT ACTUATOR CONTROLS")
        print("###############################")
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            ctrl_val = self.data.ctrl[i]
            print(f"Actuator {actuator_name} (ID {i}): control = {ctrl_val}")
