from bulletarm import env_factory
import torch
import time
import pdb
import matplotlib.pyplot as plt

import random

random.seed(42)
torch.manual_seed(42)

def save_multiple_images(images, titles=None, filename="comparison.png", cmap='gray', ncols=3, normalize=True, normalize_mode='individual'):
    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols
    
    processed_images = []
    all_data = []  
    
    for i, img_data in enumerate(images):
        if isinstance(img_data, torch.Tensor):
            img_data = img_data.detach().cpu().numpy()
        
        if img_data.shape[0] == 1:
            img_data = img_data.squeeze(0)
        
        processed_images.append(img_data)
        if normalize and normalize_mode == 'global':
            all_data.append(img_data.flatten())
    
    if normalize and normalize_mode == 'global':
        global_min = np.min(np.concatenate(all_data))
        global_max = np.max(np.concatenate(all_data))
        global_mean = np.mean(np.concatenate(all_data))
        global_std = np.std(np.concatenate(all_data))
    
    # 归一化处理
    if normalize:
        for i, img_data in enumerate(processed_images):
            if normalize_mode == 'individual':
                # 每张图像单独归一化
                img_min, img_max = img_data.min(), img_data.max()
                if img_max > img_min:  # 避免除零
                    processed_images[i] = (img_data - img_min) / (img_max - img_min)
                else:
                    processed_images[i] = np.zeros_like(img_data)
                    
            elif normalize_mode == 'global':
                # 全局归一化
                if global_max > global_min:
                    processed_images[i] = (img_data - global_min) / (global_max - global_min)
                else:
                    processed_images[i] = np.zeros_like(img_data)
                    
            elif normalize_mode == 'zero_one':
                # 强制归一化到[0,1]
                img_data = img_data.astype(np.float32)
                img_min, img_max = img_data.min(), img_data.max()
                if img_max > img_min:
                    processed_images[i] = (img_data - img_min) / (img_max - img_min)
                else:
                    processed_images[i] = np.zeros_like(img_data)
                    
            elif normalize_mode == 'standardize':
                # 标准化 (z-score)
                img_mean = img_data.mean()
                img_std = img_data.std()
                if img_std > 0:
                    processed_images[i] = (img_data - img_mean) / img_std
                else:
                    processed_images[i] = img_data - img_mean
    

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    

    for i, img_data in enumerate(processed_images):
        row = i // ncols
        col = i % ncols
        
        im = axes[row, col].imshow(img_data, cmap=cmap)
        
        # 添加标题
        title = ""
        if titles and i < len(titles):
            title = titles[i]
        
        # 添加数值范围信息
        if normalize:
            if normalize_mode == 'standardize':
                title += f"\n(μ={img_data.mean():.3f}, σ={img_data.std():.3f})"
            else:
                title += f"\n(range: {img_data.min():.3f} - {img_data.max():.3f})"
        else:
            title += f"\n(range: {img_data.min():.3f} - {img_data.max():.3f})"
        
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
        

    for i in range(n_images, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Comparison Images saved: {filename}")
    if normalize:
        print(f"Normalization mode: {normalize_mode}")

def runDemo(test_case):
  num_processes = 0
  env_config = {'render': False, 'num_objects': 1}
  env = env_factory.createEnvs(num_processes, 'object_grasping', env_config)
  env.setObjectInitMetaData(test_case)
  
  _, _, heightmap = env.reset() #takes 0.2-0.3 seconds
  env.getObjectMetaData()
  done = False
  while not done:
    action = env.getNextAction() # takes 0.0005 seconds
    env.step(action)
    break

  env.close()

  return heightmap, None
  
def runCustomDemo(test_case):
  num_processes = 0
  env_config = {'render': False, 'num_objects': 1}
  env = env_factory.createEnvs(num_processes, 'custom_object_grasping', env_config) # this is in fact a runner, not the env itself
  env.setObjectInitMetaData(test_case)

  _, _, heightmap = env.reset() #takes 0.2-0.3 seconds
  metadata = env.getObjectMetaData()
  pdb.set_trace()
  depthmap = env.getDepthmap() 
  done = False
  while not done:
    action = env.getNextAction()
    res = env.step(action)
    break

  env.close()
  """
  1. heightmap
  2. metadata
  3. action
  4. reward
  """
  return heightmap, depthmap
  
def test(idx=-1):
  test_case = [
    {
      "scale": 1.,
      "position": [[0.45, 0.0, 0.40]],
      "rotation": [[0., 0., 0.707, 0.707]], # in the order of : x y z w; it the a 90deg rotation of : "rotation": [[1, 0, 0, 0.]],
      "index": idx
    }
  ]

  print("Running block stacking demo...")
  heightmap, _ = runDemo(test_case)
  
  print("Running custom neural renderer demo...")
  neural_heightmap, neural_depthmap = runCustomDemo(test_case)

  # neural_heightmap = neural_heightmap * 0.40

  print("Displaying observations...")
  save_multiple_images([heightmap, neural_heightmap, neural_depthmap],
                      titles=['pybullet heightmap', 'pyredner heightmap', 'pyredner depthmap'],
                      filename="observation_comparison.png",
                      cmap='gray',
                      ncols=3)

  # import pdb; pdb.set_trace()
  
"""

meta data of initializing an object in the workspace:

- scale
  - datatype: float
  - range of value: [0.0, +inf)
  - description: enlarge or shrink the object by this factor, the volumne will be scaled by scale^3, the edge length will be scaled by scale. In this demo, the scaling factor on X, Y, Z will be the same. Noted that BulletArm set a upper and lower limit of the object's volume (per object), a improper scaling factor will be clipped to the upper or lower limit.

- position
  - datatype: a list of lists, each inner list contains 3 floats
  - range of value: (-inf, +inf), but should be in the workspace, otherwise cannot be found in the heightmap
  - description: the X, Y, Z position of the object in the workspace. The center of the workspace is x=0.45, y=0.0, the size of workspace is 0.4 * 0.4,  by default. Noted that you can use arbitrary value of z, but the object fall vertically if the z value is greater than highest plane underneath it, causing the REAL position does not equal to the value it initialized to be.
  
- rotation
  - datatype: a list of lists, each inner list contains 3 floats
  - range of value: [-1.0, +1.0], but should be a valid quaternion, otherwise the object will not be placed correctly, the quaternion is in the order of [x, y, z, w], where w is the real part and x, y, z are the imaginary parts.
  - description: quaternion describes the rotation of a object, with respect to X,Y,Z axes. Same as position, if the object is initialized improperly, it will fall vertically to the ground, causing the REAL rotation does not equal to the value it initialized to be.

- index
  - datatype: int
  - range of value: [0, 84]
  - description: the id of a GraspNet1B object, the source files locates in /home/weilab/tingxi/original-bulletarm/BulletArm/bulletarm/pybullet/urdf/object/GraspNet1B_object

"""

if __name__ == '__main__':
  for i in range(84):
    test(i)
    break