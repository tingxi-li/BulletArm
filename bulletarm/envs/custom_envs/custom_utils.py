import pyredner
import torch
import bulletarm
from bulletarm.pybullet.utils import constants
import os
import copy
from transforms3d import quaternions
import bulletarm.envs.configs as env_configs

config = {**env_configs.DEFAULT_CONFIG}
workspace = config['workspace']
x_shift = workspace[0].mean()
y_shift = workspace[1].mean()
z_shift = 0.0

def getMesh(meta_data, device=None, enable_gradient=False):
    if device is None:
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    index = meta_data['index']
    scale = meta_data['scale']
    rotation = meta_data['rotation']
    file_name = meta_data['file_name']
    position = torch.tensor(meta_data['position'], dtype=torch.float32, device=device)
    
    mesh = pyredner.load_obj(file_name, return_objects=True)[0]
    
    # Create transformation matrix
    quat = (rotation[3], rotation[0], rotation[1], rotation[2])
    rotation_matrix = torch.tensor(
        quaternions.quat2mat(quat), 
        dtype=torch.float32, 
        device=device
    )

    if enable_gradient:
        rotation_matrix.requires_grad = True
        position.requires_grad = True
    
    # Apply scale to rotation matrix
    transform_matrix = rotation_matrix * scale
    
    vertices = mesh.vertices.clone().float().to(device)
    
    # Transform: (Scale * Rotate) + Translate
    mesh.vertices = torch.matmul(vertices, transform_matrix.T) + position + torch.tensor([x_shift, y_shift, z_shift], dtype=torch.float32, device=device)

    return mesh, (position, rotation_matrix)


def getTrayMesh(device=None):
	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	root_dir = os.path.dirname(bulletarm.__file__)
	tray_path = os.path.join(root_dir, constants.OBJECTS_PATH, 'GraspNet1B_object/tray/tray.obj')
	tray_mesh = pyredner.load_obj(tray_path, return_objects=True)[0]
	tray_mesh.vertices = tray_mesh.vertices * 1e-3
	tray_mesh.vertices = tray_mesh.vertices + torch.tensor([x_shift, y_shift, z_shift], dtype=torch.float32, device=device)

	return tray_mesh, (None, None,)


def renderMesh(meshes, cam_pos, cam_up_vector, target_pos, fov, heightmap_size, device=None):
	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	cam_pos = torch.FloatTensor(cam_pos).to(device)
	cam_up_vector = torch.FloatTensor(cam_up_vector).to(device)
	target_pos = torch.FloatTensor(target_pos).to(device)
	fov = torch.tensor([fov], dtype=torch.float32).to(device)

	# Configure camera
	camera = pyredner.Camera(
		position=cam_pos,
		look_at=target_pos,
		up=cam_up_vector,
		fov=fov,  # in degrees
		clip_near=1e-2,  # needs to be > 0
		resolution=(heightmap_size, heightmap_size)
	)

	# Create scene and render depth
	scene = pyredner.Scene(camera=camera, objects=meshes)
	chan_list = [pyredner.channels.depth]
	depth_img = pyredner.render_generic(scene, chan_list, device=pyredner.device)

	# Convert depth to heightmap
	near = 0.09
	far = 0.010
	slope = 37821.71428571428
	intercept = - 3407.3605408838816

	depth = near * far / (far - depth_img)
	heightmap = torch.abs(depth - torch.max(depth))

	# Apply scaling and thresholding
	heightmap = heightmap * slope + intercept
	heightmap = torch.relu(heightmap)
	heightmap = torch.where(heightmap > 1.0, 6e-3, heightmap)

	return heightmap.reshape([1, heightmap_size, heightmap_size])

def getObservation(
	neural_obs, 
	action, 
	meshes, cam_pos, cam_up_vector, target_pos, fov, heightmap_size, device=None
	):

	neural_obs_copy = copy.deepcopy(nerual_obs)
	nerual_obs = renderMesh(meshes, cam_pos, cam_up_vector, target_pos, fov, heightmap_size, device)



