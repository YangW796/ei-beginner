from collections import namedtuple
import math
import time
from attrdict import AttrDict
import numpy as np
import pybullet as p

def step_simulation():
    p.stepSimulation()
    time.sleep( 1 / 240.)  


class ArmBase:
    def __init__(self,path):
        self.robot_id = p.loadURDF(
            path,
            [0, 0, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        self.joints = AttrDict()
        self.parse_joint_info()
        self.eefID = 7
        # Setup some Limit
        self.gripper_open_limit = (0.0, 0.085)
        self.ee_position_limit = ((-0.224, 0.224),(-0.724, -0.276),(1.0, 1.3))
        self.controlJoints = [
            "shoulder_pan_joint","shoulder_lift_joint",
            "elbow_joint", 
            "wrist_1_joint",
            "wrist_2_joint", 
            "wrist_3_joint",
            "finger_joint"
        ]
        self.mimicParentName = "finger_joint"
    
    # 关节信息解析 
    def parse_joint_info(self):
        numJoints = p.getNumJoints(self.robot_id)
        jointInfo = namedtuple('jointInfo', 
            ['id','name','type','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        controlJoints = [
            "shoulder_pan_joint","shoulder_lift_joint",
            "elbow_joint", 
            "wrist_1_joint",
            "wrist_2_joint", 
            "wrist_3_joint",
            "finger_joint"
        ]
        
        for i in range(numJoints):
            info = p.getJointInfo(self.robot_id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType =  jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            # 存储controllable_joint id
            controllable = True if jointName in controlJoints else False
            # 存储joint信息
            info = jointInfo(jointID,jointName,jointType,jointLowerLimit,
            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            if info.type == "REVOLUTE":  
                p.setJointMotorControl2(self.robot_id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info
    
    def enable_finger_force(self):
        p.enableJointForceTorqueSensor(self.robot_id, self.joints['left_inner_finger_pad_joint'].id)
        p.enableJointForceTorqueSensor(self.robot_id, self.joints['right_inner_finger_pad_joint'].id)
        p.changeDynamics(self.robotID, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=0.5)
        p.changeDynamics(self.robotID, self.joints['right_inner_finger_pad_joint'].id, lateralFriction=0.5)
    
    
    
    def control_gripper(self,control_mode,target_pos):
        assert control_mode == p.POSITION_CONTROL
        mimicParentName = "finger_joint"
        mimicChildren ={"right_outer_knuckle_joint": 1,
        "left_inner_knuckle_joint": 1,
        "right_inner_knuckle_joint": 1,
        "left_inner_finger_joint": -1,"right_inner_finger_joint": -1}
        parent = self.joints[mimicParentName]
        children = AttrDict((j, self.joints[j]) for j in self.joints if j in mimicChildren.keys())
        
        p.setJointMotorControl2(
            self.robot_id, parent.id, 
            control_mode, targetPosition=target_pos,
            force=parent.maxForce, maxVelocity=parent.maxVelocity
        )
            # move child joints
        for name in children:
            child = children[name]
            childPose = target_pos *  mimicChildren[child.name]
            p.setJointMotorControl2(
                self.robot_id, child.id, 
                control_mode, targetPosition=childPose,
                force=child.maxForce, maxVelocity=child.maxVelocity
            )
            
    
        
    def gripper_contact(self, force=100):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id
        contact_left = p.getContactPoints(bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robot_id, linkIndexA=right_index)

        # Check the force
        left_force = p.getJointState(self.robot_id, left_index)[2][:3]
        right_force = p.getJointState(self.robot_id, right_index)[2][:3]
        left_norm, right_norm = np.linalg.norm(left_force), np.linalg.norm(right_force)
        return left_norm > force or right_norm > force
    
    def move_gripper(self, gripper_opening_length, step = 120):
        gripper_opening_length = np.clip(gripper_opening_length, *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143) 
        for _ in range(step):
            self.control_gripper(control_mode=p.POSITION_CONTROL, target_pos=gripper_opening_angle)
            step_simulation()   

    def open_gripper(self, step= 120):
        self.move_gripper(0.085, step)

    def close_gripper(self, step= 120,check_contact=True) -> bool:
        # Get initial gripper open position
        initial_position = p.getJointState(self.robot_id, self.joints[self.mimicParentName].id)[0]
        initial_position = math.sin(0.715 - initial_position) * 0.1143 + 0.010
        for step_idx in range(1, step):
            current_target_open_length = initial_position - step_idx / step * initial_position

            self.move_gripper(current_target_open_length, 1)
            if current_target_open_length < 1e-5:
                return False

            if check_contact and self.gripper_contact():
                return True
        return False
        
    def move_away_arm(self):
        joint = self.joints['shoulder_pan_joint']
        for _ in range(200):
            p.setJointMotorControl2(
                self.robot_id, 
                joint.id, 
                p.POSITION_CONTROL,
                targetPosition=0., 
                force=joint.maxForce,
                maxVelocity=joint.maxVelocity
            )
            step_simulation()     
    
    def move_ee(self, action, max_step=500,custom_velocity=None,try_close_gripper=False):
        x, y, z, orn = action
        x = np.clip(x, *self.ee_position_limit[0])
        y = np.clip(y, *self.ee_position_limit[1])
        z = np.clip(z, *self.ee_position_limit[2])
        # set damping for robot arm and gripper
        jd = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        jd = jd * 0
        still_open_flag_ = True  # Hot fix
        for _ in range(max_step):
            # apply IK
            joint_poses = p.calculateInverseKinematics(self.robot_id, self.eefID, [x, y, z], orn,
            maxNumIterations=100, jointDamping=jd)
            for i, name in enumerate(self.controlJoints[:-1]):  # Filter out the gripper
                joint = self.joints[name]
                pose = joint_poses[i]
                # control robot end-effector
                p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                targetPosition=pose, force=joint.maxForce,
                maxVelocity=joint.maxVelocity if custom_velocity is None else custom_velocity * (i+1))

            step_simulation()
            if try_close_gripper and still_open_flag_ and not self.gripper_contact():
                still_open_flag_ = self.close_gripper(check_contact=True)
            
            # Check xyz and rpy error
            real_xyz, real_xyzw = p.getLinkState(self.robot_id, self.eefID)[0:2]
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            real_roll, real_pitch, real_yaw = p.getEulerFromQuaternion(real_xyzw)
            if np.linalg.norm(np.array((x, y, z)) - real_xyz) < 0.001 \
                    and np.abs((roll - real_roll, pitch - real_pitch, yaw - real_yaw)).sum() < 0.001:
                return True, (real_xyz, real_xyzw)

        # raise FailToReachTargetError
        print('Failed to reach the target')
        return False, p.getLinkState(self.robot_id, self.eefID)[0:2]
    
    ### 重置
    def reset_robot(self):
        user_parameters = (-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,-1.5707970583733368, 0.0009377758247187636, 0.085)
        for _ in range(100):
            for i, name in enumerate(self.controlJoints):
                if i == 6:
                    self.control_gripper(p.POSITION_CONTROL, user_parameters[i])
                    break
                
                joint = self.joints[name]
                # control robot joints
                p.setJointMotorControl2(
                    self.robot_id, joint.id,
                    p.POSITION_CONTROL,targetPosition=user_parameters[i], force=joint.maxForce,maxVelocity=joint.maxVelocity
                )
                step_simulation()

        
    