import copy
import cv2 as cv
import mujoco
import numpy as np


class Camera:
    def __init__(self, model, data, width=200, height=200, camera_name=""):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.renderer = mujoco.Renderer(model, width=width, height=height)

        # Camera info
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.cam_matrix = None
        self.cam_rot_mat = None
        self.cam_pos = None
        self.cam_init = False

    def get_image_data(self, show=False):
        """获取 RGB 和 Depth 图像（单位为米）"""
        # Render RGB
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb = self.renderer.render()

        # Render Depth
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        if show:
            cv.imshow("rgb", cv.cvtColor(rgb, cv.COLOR_BGR2RGB))
            depth_vis = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-5)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            depth_vis = cv.applyColorMap(depth_vis, cv.COLORMAP_JET)  # 上色
            cv.imshow("depth", depth_vis)
            cv.waitKey(1000)

        # Flip for visual consistency
        rgb = np.array(np.fliplr(np.flipud(rgb)))
        depth = np.array(np.fliplr(np.flipud(depth)))
        depth = self._depth_2_meters(depth)

        return rgb, depth

    def _depth_2_meters(self, depth):
        """MuJoCo 2.3+ 直接输出深度为米单位，这里只做 clean 和 clip"""
        depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(depth_clean, 0.0, 5.0)  # Optional: clip to max 5m

    def create_camera_data(self):
        """初始化相机内参和外参"""
        cam_id = self.camera_id
        fovy = self.model.cam_fovy[cam_id]
        f = 0.5 * self.height / np.tan(fovy * np.pi / 360)

        self.cam_matrix = np.array([[f, 0, self.width / 2],
                                    [0, f, self.height / 2],
                                    [0, 0, 1]])

        self.cam_rot_mat = np.reshape(self.model.cam_mat0[cam_id], (3, 3))
        self.cam_pos = np.array(self.model.cam_pos0[cam_id])
        self.cam_init = True

    def world_2_pixel(self, world_coordinate):
        """
        将世界坐标转换为图像像素坐标
        """
        if not self.cam_init:
            self.create_camera_data()

        wc = np.array(world_coordinate)
        relative_pos = wc - self.cam_pos
        hom_pixel = self.cam_matrix @ (self.cam_rot_mat @ relative_pos)
        pixel = hom_pixel[:2] / hom_pixel[2]

        return int(np.round(pixel[0])), int(np.round(pixel[1]))

    def pixel_2_world(self, pixel_x, pixel_y, depth):
        """
        将像素坐标 + 深度转换为世界坐标
        """
        if not self.cam_init:
            self.create_camera_data()

        # Create homogeneous coordinate in camera space
        pixel_coord = np.array([pixel_x, pixel_y, 1])
        pos_c = np.linalg.inv(self.cam_matrix) @ (pixel_coord * depth)
        pos_w = self.cam_rot_mat.T @ pos_c + self.cam_pos

        return pos_w
