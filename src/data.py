import glob
import time
import numpy as np
import pybullet as p
OBJECT_INIT_HEIGHT = 1.05
def step_simulation():
    p.stepSimulation()
   #time.sleep( 1 / 240.)  

class Models:
    def load_objects(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        return NotImplementedError
    
class YCBModels(Models):
    def __init__(self,path):
        self.files=glob.glob(path)
        self.visual_shape_ids=[]
        self.collision_shape_ids=[]
        self.obj_ids=[]
    
    def load_objects(self,num):
        self.visual_shape_ids=[]
        self.collision_shape_ids=[]
        self.obj_ids=[]
        shift = [0, 0, 0]
        mesh_scale = [1, 1, 1]
        assert(num<=len(self.files))
        for i,filename in enumerate(self.files):
            i=i+1
            if i>num: break
            print(f"Loading {i}:{filename}\n")
            col_shape_id=p.createCollisionShape(shapeType=p.GEOM_MESH,fileName=filename)
            vis_shape_id=p.createVisualShape(shapeType=p.GEOM_MESH,fileName=filename)
            self.collision_shape_ids.append(col_shape_id)
            self.visual_shape_ids.append(vis_shape_id)
            
            obj_handle = p.createMultiBody(
                baseMass=0.2,
                baseInertialFramePosition=[0, 0, 0],
                baseCollisionShapeIndex=col_shape_id,
                baseVisualShapeIndex=vis_shape_id,
                basePosition=(np.clip(np.random.normal(0, 0.005), -0.2, 0.2),np.clip(np.random.normal(0, 0.005) - 0.5, -0.7, -0.3),OBJECT_INIT_HEIGHT),# useMaximalCoordinates=True,
                baseOrientation=p.getQuaternionFromEuler((np.random.uniform(-np.pi, np.pi),np.random.uniform(0, np.pi),np.random.uniform(-np.pi, np.pi)))
            )
            
            p.changeDynamics(obj_handle, -1, lateralFriction=1, rollingFriction=0.01, spinningFriction=0.001,restitution=0.01)
            self.obj_ids.append(obj_handle)
            self.wait_until_still()
        return self.obj_ids   
            
    def is_still(self,handle):
        still_eps = 1e-3
        lin_vel, ang_vel = p.getBaseVelocity(handle)
        return np.abs(lin_vel).sum() + np.abs(ang_vel).sum() < still_eps

    def wait_until_still(self, max_wait_epochs=1000,):
        for _ in range(max_wait_epochs):
            step_simulation()
            if np.all(list(self.is_still(handle) for handle in self.obj_ids)):
                return
        print(f'Warning: Not still after MAX_WAIT_EPOCHS ={max_wait_epochs}.' )
    
    
    def __len__(self):
        return len(self.collision_shape_ids)

    def __getitem__(self, idx):
        print(len(self.visual_shape_ids))
        return self.visual_shape_ids[idx], self.collision_shape_ids[idx]

            