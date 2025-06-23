from bulletarm import env_factory
import torch
import time
import pdb
import matplotlib.pyplot as plt



def save_multiple_images(images, titles=None, filename="comparison.png", cmap='gray', ncols=3):
    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_data in enumerate(images):
        row = i // ncols
        col = i % ncols
        
        if isinstance(img_data, torch.Tensor):
            img_data = img_data.detach().cpu().numpy()
        
        if img_data.shape[0] == 1:
            img_data = img_data.squeeze(0)
        
        axes[row, col].imshow(img_data, cmap=cmap)
        if titles and i < len(titles):
            axes[row, col].set_title(titles[i])
        axes[row, col].axis('off')
    
    for i in range(n_images, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()  
    print(f"Comparison Images saved: {filename}")


def runDemo(test_case):
  num_processes = 0
  env_config = {'render': False, 'num_objects': 1}
  env = env_factory.createEnvs(num_processes, 'object_grasping', env_config)
  env.setObjectInitMetaData(test_case)
  
  old_obs = env.reset() #takes 0.2-0.3 seconds
  old_obj_metadata = env.getObjectMetaData()

  done = False
  while not done:
    action = env.getNextAction() # takes 0.0005 seconds
    obs, reward, done = env.step(action)
  obj_metadata = env.getObjectMetaData()

  env.close()

  return old_obs, old_obj_metadata, obs, obj_metadata, reward
  
def runCustomDemo(test_case):
  num_processes = 0
  env_config = {'render': False, 'num_objects': 1}
  env = env_factory.createEnvs(num_processes, 'custom_object_grasping', env_config) # this is in fact a runner, not the env itself
  env.setObjectInitMetaData(test_case)
  old_obs = env.reset() #takes 0.2-0.3 seconds
  old_obj_metadata = env.getObjectMetaData()
  done = False
  while not done:
    action = env.getNextAction()
    obs, reward, done = env.step(action)
  obj_metadata = env.getObjectMetaData()

  env.close()

  return old_obs, old_obj_metadata, obs, obj_metadata, reward

def test(idx=0):
  test_case = [
    {
      "scale": 1,
      "position": [[0.45, 0.0, 0.40]],
      "rotation": [[0., 0., 0.]],
      "index": idx
    }
  ]

  print("Running block stacking demo...")
  (_, _, old_obs), old_obj_metadata, (_, _, obs), obj_metadata, reward = runDemo(test_case)
  
  print("Running custom neural renderer demo...")
  (_, _, old_neural_renderer_obs), old_neural_obj_metadata, (_, _, neural_obs), obj_neural_metadata, neural_reward = runCustomDemo(test_case)

  print("Displaying observations...")
  save_multiple_images([old_obs, old_neural_renderer_obs],
                      titles=['Observation from BulletArm', 'Observation from Custom Neural Renderer'],
                      filename="observation_comparison.png",
                      cmap='gray',
                      ncols=2)
  
if __name__ == '__main__':
  for i in range(84):
    test(i)
