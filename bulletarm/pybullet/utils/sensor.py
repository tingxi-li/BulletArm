import pybullet as pb
import numpy as np
from transforms3d import quaternions
import torch
import pyredner
import numpy.random as npr
import sys
import os
from pathlib import Path
import bulletarm.pybullet.utils.constants as constants
pyredner.set_print_timing(False)

class Sensor(object):
  def __init__(self, cam_pos, cam_up_vector, target_pos, target_size, near, far):
    self.view_matrix = pb.computeViewMatrix(
      cameraEyePosition=cam_pos,
      cameraUpVector=cam_up_vector,
      cameraTargetPosition=target_pos,
    )

    self.near = near
    self.far = far
    self.fov = np.degrees(2 * np.arctan((target_size / 2) / self.far))
    self.proj_matrix = pb.computeProjectionMatrixFOV(self.fov, 1, self.near, self.far)

  def setCamMatrix(self, cam_pos, cam_up_vector, target_pos):
    self.view_matrix = pb.computeViewMatrix(
      cameraEyePosition=[cam_pos[0], cam_pos[1], cam_pos[2]],
      cameraUpVector=cam_up_vector,
      cameraTargetPosition=target_pos,
    )
    self.proj_matrix = pb.computeProjectionMatrixFOV(70, 1, 0.001, 0.3)

  def getHeightmap(self, size):
    image_arr = pb.getCameraImage(width=size, height=size,
                                  viewMatrix=self.view_matrix,
                                  projectionMatrix=self.proj_matrix,
                                  renderer=pb.ER_TINY_RENDERER)
    depth_img = np.array(image_arr[3])
    depth = self.far * self.near / (self.far - (self.far - self.near) * depth_img)

    return np.abs(depth - np.max(depth)).reshape(size, size)

  def getRGBImg(self, size):
    image_arr = pb.getCameraImage(width=size, height=size,
                                  viewMatrix=self.view_matrix,
                                  projectionMatrix=self.proj_matrix,
                                  renderer=pb.ER_TINY_RENDERER)
    rgb_img = np.moveaxis(image_arr[2][:, :, :3], 2, 0) / 255
    return rgb_img

  def getDepthImg(self, size):
    image_arr = pb.getCameraImage(width=size, height=size,
                                  viewMatrix=self.view_matrix,
                                  projectionMatrix=self.proj_matrix,
                                  renderer=pb.ER_TINY_RENDERER)
    depth_img = np.array(image_arr[3])
    depth = self.far * self.near / (self.far - (self.far - self.near) * depth_img)
    return depth.reshape(size, size)

  def getPointCloud(self, size, to_numpy=True):
    image_arr = pb.getCameraImage(width=size, height=size,
                                  viewMatrix=self.view_matrix,
                                  projectionMatrix=self.proj_matrix,
                                  renderer=pb.ER_TINY_RENDERER)
    depthImg = np.asarray(image_arr[3])

    # https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
    projectionMatrix = np.asarray(self.proj_matrix).reshape([4,4],order='F')
    viewMatrix = np.asarray(self.view_matrix).reshape([4,4],order='F')
    tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
    pixel_pos = np.mgrid[0:size, 0:size]
    pixel_pos = pixel_pos/(size/2) - 1
    pixel_pos = np.moveaxis(pixel_pos, 1, 2)
    pixel_pos[1] = -pixel_pos[1]
    zs = 2*depthImg.reshape(1, size, size) - 1
    pixel_pos = np.concatenate((pixel_pos, zs))
    pixel_pos = pixel_pos.reshape(3, -1)
    augment = np.ones((1, pixel_pos.shape[1]))
    pixel_pos = np.concatenate((pixel_pos, augment), axis=0)
    position = np.matmul(tran_pix_world, pixel_pos)
    pc = position / position[3]
    points = pc.T[:, :3]

    # if to_numpy:
      # points = np.asnumpy(points)
    return points

class SensorPyRedner(object):
  def __init__(self, config, device=None):
    if device is None:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    self.seed = config['seed']
    npr.seed(self.seed)
    self.workspace = config['workspace']
    self.shift = torch.tensor([self.workspace[0].mean(), self.workspace[1].mean(), 0.0], dtype=torch.float32).to(self.device)
    
    self.heightmap_size = config['obs_size']
    self.in_hand_size = config['in_hand_size']
    self.in_hand_mode = config['in_hand_mode']
    
    ws_size = max(self.workspace[0][1] - self.workspace[0][0], self.workspace[1][1] - self.workspace[1][0])
    self.cam_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 10]
    self.target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
    self.cam_up_vector = [-1, 0, 0]
    self.fov = np.degrees(2 * np.arctan((ws_size / 2) / self.cam_pos[2]))
    
    self.cam_pos = torch.tensor(self.cam_pos, dtype=torch.float32, device=self.device)
    self.cam_up_vector = torch.tensor(self.cam_up_vector, dtype=torch.float32, device=self.device)
    self.target_pos = torch.tensor(self.target_pos, dtype=torch.float32, device=self.device)
    self.fov = torch.tensor([self.fov], dtype=torch.float32, device=self.device)
    
    self.camera = pyredner.Camera(
      position=self.cam_pos,
      look_at=self.target_pos,
      up=self.cam_up_vector,
      fov=self.fov,  # in degrees
      clip_near=1e-2,  # needs to be > 0
      resolution=(self.heightmap_size, self.heightmap_size),
    )

  def getMeshes(self, meta_data):
      
    index = meta_data['index']
    scale = meta_data['scale']
    rotation = meta_data['rotation']
    file_name = meta_data['file_name']
    position = torch.tensor(meta_data['position'], dtype=torch.float32, device=self.device)
    
    mesh = pyredner.load_obj(file_name, return_objects=True)[0]

    quat = (rotation[3], rotation[0], rotation[1], rotation[2])
    rotation_matrix = torch.tensor(
        quaternions.quat2mat(quat), 
        dtype=torch.float32, 
        device=self.device
    )

    # if enable_gradient:
    #     rotation_matrix.requires_grad = True
    #     position.requires_grad = True
    
    transform_matrix = rotation_matrix * scale
    vertices = mesh.vertices.clone().float().to(self.device)
    mesh.vertices = torch.matmul(vertices, transform_matrix.T) + position #+ self.shift

    return mesh, (position, rotation_matrix)

  def getTrayMesh(self):
    root_dir = Path(__file__).parent.parent.parent
    tray_path = os.path.join(root_dir, constants.OBJECTS_PATH, 'GraspNet1B_object/tray/tray_textured.obj')
    tray_mesh = pyredner.load_obj(tray_path, return_objects=True)
    quat = (0, 0, 0, 1)  # Identity quaternion
    rotation_matrix = torch.tensor(
        quaternions.quat2mat(quat), 
        dtype=torch.float32, 
        device=self.device
    )
    x_y_z_scale = torch.tensor([0.344, 0.344, 0.135], dtype=torch.float32, device=self.device)

    transformed_mesh = []
    for tray_part in tray_mesh:
      tmp = tray_part.vertices.clone().float().to(self.device)
      rotated_tmp = torch.matmul(tmp, rotation_matrix.T)
      scaled_tmp = rotated_tmp * x_y_z_scale
      shifted_tmp = scaled_tmp + self.shift 
      tray_part.vertices = shifted_tmp
      transformed_mesh.append(tray_part)

    return transformed_mesh

  def getDepthmap(self, meshes):
    scene = pyredner.Scene(camera=self.camera, objects=meshes)
    chan_list = [pyredner.channels.depth]
    depth_img = pyredner.render_generic(scene, chan_list, device=self.device)

    return depth_img.reshape([self.heightmap_size, self.heightmap_size])
    
  # def _getHeightmap(self, meshes):
  #   # Implement heightmap generation logic here
  #   scene = pyredner.Scene(camera=self.camera, objects=meshes)
  #   chan_list = [pyredner.channels.depth]
  #   depth_img = pyredner.render_generic(scene, chan_list, device=self.device)
    
  #   near = 0.09
  #   far = 0.010
  #   slope = 37821.71428571428
  #   intercept = - 3407.3605408838816

  #   depth = near * far / (far - depth_img)
  #   heightmap = torch.abs(depth - torch.max(depth))

  #   # Apply scaling and thresholding
  #   heightmap = heightmap * slope + intercept
  #   heightmap = torch.relu(heightmap)
  #   heightmap = torch.where(heightmap > 1.0, torch.tensor(6e-3).to(self.device), heightmap)
    
  #   heightmap =  heightmap.reshape([self.heightmap_size, self.heightmap_size])
  #   return heightmap

  def getHeightmap(self, meshes):
    scene = pyredner.Scene(camera=self.camera, objects=meshes)
    chan_list = [pyredner.channels.position, pyredner.channels.alpha]
    rendered_output = pyredner.render_generic(scene, chan_list, device=self.device)
    # Extract world positions and alpha mask
    # rendered_output shape: [heightmap_size, heightmap_size, 4]
    # Channels 0-2: world position (X, Y, Z)
    # Channel 3: alpha (foreground mask)
    world_positions = rendered_output[:, :, :3]  # [H, W, 3]
    alpha_mask = rendered_output[:, :, 3]        # [H, W]
    heightmap = world_positions[:, :, 2] # Z component
    valid_mask = alpha_mask > 0.5
  
    if torch.sum(valid_mask) > 0:
        # Find minimum height in the scene to use for background
        min_height = torch.min(heightmap[valid_mask])
        
        # Create final heightmap
        final_heightmap = heightmap.clone()
        final_heightmap[~valid_mask] = min_height
    else:
        # If no valid pixels, return zeros
        final_heightmap = torch.zeros_like(heightmap)
    
    return final_heightmap.reshape([self.heightmap_size, self.heightmap_size])
  