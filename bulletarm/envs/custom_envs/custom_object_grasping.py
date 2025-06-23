import os
import math
import glob
import numpy as np
import bulletarm
import csv
from bulletarm.envs.base_env import BaseEnv
from bulletarm.envs.base_env_pyredner import BaseEnvPyRedner
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.pybullet.equipments.tray import Tray
from scipy.ndimage.interpolation import rotate
import pybullet as pb
import bulletarm.envs.configs as env_configs
import torch
import inspect
import json
import pyredner
from . import custom_utils

class ObjectGrasping(BaseEnvPyRedner):
    def __init__(self, config):
        # env specific parameters
        if 'object_scale_range' not in config:
            # config['object_scale_range'] = {**env_configs.DEFAULT_CONFIG, **config}['object_scale_range']
            config['object_scale_range'] = [1.,1.]
        if 'num_objects' not in config:
            config['num_objects'] = 15
        if 'max_steps' not in config:
            config['max_steps'] = 50
        if 'object_index' not in config:
            config["object_index"] = -1
        config['adjust_gripper_after_lift'] = True
        config['min_object_distance'] = 0.
        config['min_boarder_padding'] = 0.15
        super(ObjectGrasping, self).__init__(config)
        self.object_init_z = 0.1
        self.obj_grasped = 0
        self.tray = Tray()
        self.exhibit_env_obj = False
        # self.exhibit_env_obj = True
        self.bin_size = self.workspace_size - 0.1
        self.gripper_depth = 0.04
        self.gripper_clearance = 0.01
        self.initialized = False
        self.object_index = config["object_index"]
        
    def initialize(self):
        super().initialize()
        self.tray.initialize(pos=[self.workspace[0].mean(), self.workspace[1].mean(), 0],
                             size=[self.bin_size + 0.03, self.bin_size + 0.03, 0.1])
        self.initialized = True
        
    def _decodeAction(self, action):
        """
    decode input action base on self.action_sequence
    Args:
      action: action tensor

    Returns: motion_primative, x, y, z, rot

    """
        primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a),
                                                          ['p', 'x', 'y', 'z', 'r'])
        motion_primative = action[primative_idx] if primative_idx != -1 else 0
        if self.action_sequence.count('r') <= 1:
            rz = action[rot_idx] if rot_idx != -1 else 0
            ry = 0
            rx = 0
        else:
            raise NotImplementedError
        x = action[x_idx]
        y = action[y_idx]
        # x += (self.workspace[0, 0] + self.workspace[0, 1]) / 2
        # y += (self.workspace[1, 0] + self.workspace[1, 1]) / 2
        if z_idx != -1:
            z = action[z_idx]
        else:
            z = self.getPatch_z(x, y, rz)

        rot = (rx, ry, rz)

        return motion_primative, x, y, z, rot
    

    def _getPixelsFromPos(self, x, y):
        row_pixel, col_pixel = super()._getPixelsFromPos(x, y)
        row_pixel = min(row_pixel, self.heightmap_size - self.in_hand_size / 2 - 1)
        row_pixel = max(row_pixel, self.in_hand_size / 2)
        col_pixel = min(col_pixel, self.heightmap_size - self.in_hand_size / 2 - 1)
        col_pixel = max(col_pixel, self.in_hand_size / 2)
        return row_pixel, col_pixel

    def getPatch_z(self, x, y, rz, z=None):
        """
        get the image patch in heightmap, centered at center_pixel, rotated by rz
        :param obs:
        :param center_pixel
        :param rz:
        :return: safe z
        """
        row_pixel, col_pixel = self._getPixelsFromPos(x, y)
        # local_region is as large as ih_img
        local_region = self.heightmap[int(row_pixel - self.in_hand_size / 2): int(row_pixel + self.in_hand_size / 2),
                       int(col_pixel - self.in_hand_size / 2): int(col_pixel + self.in_hand_size / 2)]
        if isinstance(local_region, torch.Tensor):
            local_region = local_region.cpu().numpy()
        # if isinstance(rz, torch.Tensor):
        #     rz = rz.cpu().numpy()
        local_region = rotate(local_region, angle=-rz * 180 / np.pi, reshape=False)
        patch = local_region[int(self.in_hand_size / 2 - 16):int(self.in_hand_size / 2 + 16),
                int(self.in_hand_size / 2 - 4):int(self.in_hand_size / 2 + 4)]
        if z is None:
            edge = patch.copy()
            edge[5:-5] = 0
            # safe_z_pos = max(np.mean(patch.flatten()[(-patch).flatten().argsort()[2:12]]) - self.gripper_depth,
            #                  np.mean(edge.flatten()[(-edge).flatten().argsort()[:6]]) - 0.005)
            safe_z_pos = np.mean(patch.flatten()[(-patch).flatten().argsort()[2:12]]) - self.gripper_depth
            safe_z_pos += self.workspace[2, 0]
        else:
            safe_z_pos = np.mean(patch.flatten()[(-patch).flatten().argsort()[2:12]]) + z

        # use clearance to prevent gripper colliding with ground
        safe_z_pos = max(safe_z_pos, self.workspace[2, 0] + self.gripper_clearance)
        safe_z_pos = min(safe_z_pos, self.workspace[2, 1])
        # assert safe_z_pos >= self.workspace[2][0] + self.gripper_clear
        assert self.workspace[2][0] <= safe_z_pos <= self.workspace[2][1]

        return safe_z_pos

    def _checkPerfectGrasp(self, x, y, z, rot, objects):
        return True
    
    def step(self, action):
        pre_obj_grasped = self.obj_grasped
        self.takeAction(action)
        self.wait(100)
        
        obs = self._getObservation(action)
        done = self._checkTermination()
        reward = 1.0 if self.obj_grasped > pre_obj_grasped else 0.0

        self.current_episode_steps += 1

        return obs, reward, True
    
    def isSimValid(self):
        for obj in self.objects:
            p = obj.getPosition()
            if not self.check_random_obj_valid and self.object_types[obj] == constants.RANDOM:
                continue
            if obj.getPosition()[2] >= 0.35 or self._isObjectHeld(obj):
                continue
            if self.workspace_check == 'point':
                if not self._isPointInWorkspace(p):
                    return False
            else:
                if not self._isObjectWithinWorkspace(obj):
                    return False
            if self.pos_candidate is not None:
                if np.abs(self.pos_candidate[0] - p[0]).min() > 0.02 or np.abs(
                    self.pos_candidate[1] - p[1]).min() > 0.02:
                    return False
        return True
    

    def setShapesConfig(self, shapes_config=None,):
        if shapes_config is None:
            shapes_config = []
            for _ in range(self.num_obj):
                x = (np.random.rand() - 0.5) * 0.1
                x += self.workspace[0].mean()
                y = (np.random.rand() - 0.5) * 0.1
                y += self.workspace[1].mean()
                _config = {
                    "xyz": [x, y, 0.40],
                    "rot": None,
                    "model_id": -1,
                    "scale": None, # noted that for object grasping, each object has a DIFFERENT max scale, anything larger than this will be clipped
                }
                shapes_config.append(_config)
        else:
            assert len(shapes_config) == self.num_obj, "shapes config must have the same length as num_obj"
        pass
    
    # def reset(self, pos_list=None):
    #     if pos_list is None or len(pos_list) == 0:
    #         pos_list = []
    #         for _ in range(self.num_obj):
    #             x = (np.random.rand() - 0.5) * 0.1
    #             x += self.workspace[0].mean()
    #             y = (np.random.rand() - 0.5) * 0.1
    #             y += self.workspace[1].mean()
    #             pos_list.append([x, y, 0.40])
    #     else:
    #         assert len(pos_list) == self.num_obj, "pos_list must have the same length as num_obj"
            
    #     if not self.initialized or self.obj_grasped == self.num_obj or self.current_episode_steps > self.max_steps or not self.isSimValid():
    #         while True:
    #             self.resetPybulletWorkspace()
    #             try:
    #                 if not self.exhibit_env_obj:
    #                     for i in range(self.num_obj):
    #                         randpos = pos_list[i]
                            
    #                         obj = self._generateShapes(constants.GRASP_NET_OBJ, self.num_obj,
    #                                                    random_orientation=self.random_orientation,
    #                                                    pos=[randpos], padding=self.min_boarder_padding,
    #                                                    min_distance=self.min_object_distance, model_id=-1)
    #                         pb.changeDynamics(obj[0].object_id, -1, lateralFriction=0.6)
    #                         self.wait(10)
    #                 elif self.exhibit_env_obj:  # exhibit all random objects in this environment
    #                     raise NotImplementedError("This part does not need to be implemented for reset_requires_grad")
                    
    #             except NoValidPositionException:
    #                 continue
    #             else:
    #                 break
    #         self.wait(200)
    #         self.obj_grasped = 0
            
    #     return self._getObservation()
    
    def reset(self):
        
        if not self.initialized or self.obj_grasped == self.num_obj or self.current_episode_steps > self.max_steps or not self.isSimValid():
            while True:
                self.resetPybulletWorkspace()
                try:
                    if not self.exhibit_env_obj:
                        for i in range(self.num_obj):
                            try:
                                obj_info = self.object_init_metadata[i]
                            except IndexError:
                                raise ValueError("use setObjectInitMetaData() to set object_init_metadata before reset")
                            position = obj_info["position"]
                            rotation = obj_info["rotation"]
                            scale = obj_info["scale"]
                            model_id = obj_info["index"]
                            obj = self._generateShapes(shape_type=constants.GRASP_NET_OBJ, 
                                                       num_shapes=1,
                                                       random_orientation=self.random_orientation,
                                                       pos=position, 
                                                       rot=rotation,
                                                       scale=scale,
                                                       padding=self.min_boarder_padding,
                                                       min_distance=self.min_object_distance, 
                                                       model_id=model_id)
                            pb.changeDynamics(obj[0].object_id, -1, lateralFriction=0.6)
                            self.wait(10)
                    elif self.exhibit_env_obj:  # exhibit all random objects in this environment
                        raise NotImplementedError("This part does not need to be implemented for reset_requires_grad")
                    
                except NoValidPositionException:
                    continue
                else:
                    break
            self.wait(200)
            self.obj_grasped = 0
            
        return self._getObservation()
    
    
    def isObjInBox(self, obj_pos, tray_pos, tray_size):
        tray_range = self.tray_range(tray_pos, tray_size)
        return tray_range[0][0] < obj_pos[0] < tray_range[0][1] and tray_range[1][0] < obj_pos[1] < tray_range[1][1]

    @staticmethod
    def tray_range(tray_pos, tray_size):
        return np.array([[tray_pos[0] - tray_size[0] / 2, tray_pos[0] + tray_size[0] / 2],
                         [tray_pos[1] - tray_size[1] / 2, tray_pos[1] + tray_size[1] / 2]])

    def InBoxObj(self, tray_pos, tray_size):
        obj_list = []
        for obj in self.objects:
            if self.isObjInBox(obj.getPosition(), tray_pos, tray_size):
                obj_list.append(obj)
        return obj_list

    def _checkTermination(self):
        ''''''
        for obj in self.objects:
            # if self._isObjectHeld(obj):
            #   self.obj_grasped += 1
            #   self._removeObject(obj)
            #   if self.obj_grasped == self.num_obj:
            #     return True
            #   return False
            if obj.getPosition()[2] >= 0.35 or self._isObjectHeld(obj):
                # ZXP getPos z > threshold is more robust than _isObjectHeld()
                self.obj_grasped += 1
                self._removeObject(obj)
                if self.obj_grasped == self.num_obj or len(self.objects) == 0:
                    return True
                return False
        return False
    
    def _getObservation(self, action=None):
        tray_mesh = self.sensor_pyredner.getTrayMesh()
        meshes = [tray_mesh]
        state, in_hand, obs = super(ObjectGrasping, self)._getObservation(meshes)
        
        return 0, torch.zeros_like(in_hand), obs
    
    def setObjectInitMetaData(self, object_init_metadata=None):
        """ IMPORTANT: This function should be called no more than once per environment reset."""
        
        if object_init_metadata is None:
            object_init_metadata = []
            for obj in range(self.num_obj):
                _x = (torch.rand(()) - 0.5) * 0.1
                _y = (torch.rand(()) - 0.5) * 0.1
                _x = _x + self.workspace[0].mean()
                _y = _y + self.workspace[1].mean()
                
                _info = {
                    "scale": None,
                    "position": [[_x.item(), _y.item(), 0.40]],
                    "rotation": None, # it has to be in [[]], like position
                    "index": -1,
                }
                object_init_metadata.append(_info)
        else:
            pass
        
        self.object_init_metadata = object_init_metadata

def createCustomObjectGrasping(config):
    return ObjectGrasping(config)