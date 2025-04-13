import pybullet as p
import math

# 欧拉角，单位是弧度
euler_angles = [math.radians(30), math.radians(45), math.radians(60)]

# 获取四元数
quaternion = p.getQuaternionFromEuler(euler_angles)
print(quaternion)