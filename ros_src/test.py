#!/usr/bin/env python
import rospy
import actionlib
from moveit_msgs.msg import MoveGroupAction, MoveGroupGoal
from geometry_msgs.msg import PoseStamped

class GraspActionServer:
    def __init__(self):
        self.server = actionlib.SimpleActionServer('grasp_action', GraspAction, self.execute, False)
        self.server.start()
        
        # MoveIt动作客户端
        self.moveit_client = actionlib.SimpleActionClient('move_group', MoveGroupAction)
        self.moveit_client.wait_for_server()
        
    def execute(self, goal):
        # 1. 移动到预抓取位置
        pre_grasp_pose = self.create_pre_grasp_pose(goal.object_pose)
        if not self.move_to_pose(pre_grasp_pose):
            self.server.set_aborted()
            return
            
        # 2. 执行抓取动作
        if not self.perform_grasp():
            self.server.set_aborted()
            return
            
        # 3. 移动到目标位置
        if not self.move_to_goal(goal.target_pose):
            self.server.set_aborted()
            return
            
        # 4. 释放物体
        self.release_object()
        
        self.server.set_succeeded()
        
    def create_pre_grasp_pose(self, object_pose):
        # 根据物体位姿计算预抓取位姿
        pre_grasp = PoseStamped()
        pre_grasp.header.frame_id = "base_link"
        pre_grasp.pose.position.x = object_pose.position.x - 0.05  # 在物体前方5cm
        pre_grasp.pose.position.y = object_pose.position.y
        pre_grasp.pose.position.z = object_pose.position.z
        pre_grasp.pose.orientation = object_pose.orientation
        return pre_grasp
        
    def move_to_pose(self, pose):
        goal = MoveGroupGoal()
        goal.request.workspace_parameters.header.frame_id = "base_link"
        goal.request.goal_constraints.append(self.create_pose_constraint(pose))
        goal.planning_options.plan_only = False
        
        self.moveit_client.send_goal(goal)
        self.moveit_client.wait_for_result()
        
        return self.moveit_client.get_result().error_code.val == MoveItErrorCodes.SUCCESS
        
    def perform_grasp(self):
        # 控制夹爪闭合
        rospy.loginfo("Performing grasp...")
        # 这里添加夹爪控制代码
        return True
        
    def release_object(self):
        # 控制夹爪打开
        rospy.loginfo("Releasing object...")
        # 这里添加夹爪控制代码