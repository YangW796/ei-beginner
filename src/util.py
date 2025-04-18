import numpy as np
import pybullet as p

# Copy 
class Camera_:
    def __init__(self, cam_pos, near, far, size, fov):
        self.x, self.y, self.z = cam_pos
        self.size=size
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.view_matrix = p.computeViewMatrix([self.x, self.y, self.z],
                                               [self.x - 1e-5, self.y, 0],
                                               [-1, 0, 0])
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)

        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)

    def rgbd_2_world(self, w, h, d):
        x = (2 * w - self.width) / self.width
        y = -(2 * h - self.height) / self.height
        z = 2 * d - 1
        pix_pos = np.array((x, y, z, 1))
        position = self.tran_pix_world @ pix_pos
        position /= position[3]

        return position[:3]

    def shot(self):
        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = p.getCameraImage(self.width, self.height,self.view_matrix, self.projection_matrix,)
        return rgb, depth, seg

class Camera:
    # 假设这是您的相机类实现
    def __init__(self, position, near, far, resolution, fov):
        self.position = position
        self.near = near
        self.far = far
        self.resolution = resolution
        self.fov = fov
    
    def shot(self):
        x,y,_z=self.position
        view_matrix = p.computeViewMatrix(
            self.position,
            [x - 1e-5, y, 0],
            [-1, 0, 0]
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.resolution[0]/self.resolution[1],
            nearVal=self.near,
            farVal=self.far
        )
        _, _, rgb, depth, seg = p.getCameraImage(
            width=self.resolution[0],
            height=self.resolution[1],
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            shadow = False, 
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        return rgb, depth, seg
    
    def rgbd_2_world(self, u, v, depth):
        # 实现像素坐标到世界坐标的转换
        # 简化实现，实际应根据相机参数计算
        x = (u - self.resolution[0]/2) * depth / (self.resolution[0]/2)
        y = (v - self.resolution[1]/2) * depth / (self.resolution[1]/2)
        return x, y, depth