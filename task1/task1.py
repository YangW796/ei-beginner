import pybullet as p
import time
import pybullet_data

# 连接到物理仿真环境
p.connect(p.GUI)  # GUI 模式显示仿真
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 加载一些标准的模型路径

# 加载机械臂模型 (例如UR5)
robot_id = p.loadURDF("/home/wy/Document/eibeginner/model/ur5-bullet/UR5/ur_e_description/urdf/ur5e.urdf", useFixedBase=True)

# 设置仿真步长
p.setTimeStep(0.01)

# 获取机械臂的关节数量
num_joints = p.getNumJoints(robot_id)
print(f"Mechanical Arm has {num_joints} joints.")

# 控制机械臂运动：例如设置第一个关节的位置
joint_position = [0.0] * num_joints
joint_position[0] = 0.5  # 控制第一个关节旋转0.5弧度

# 设置关节位置（通过设置每个关节的目标位置）
p.setJointMotorControlArray(robot_id, range(num_joints), p.POSITION_CONTROL, targetPositions=joint_position)

# 进行仿真
for _ in range(1000):
    p.stepSimulation()  # 进行一步仿真
    time.sleep(0.01)  # 控制仿真速度

# 断开连接
p.disconnect()
