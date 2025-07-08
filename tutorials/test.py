
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Force PyRedner to use CPU rendering to avoid CUDA context errors
import pyredner
pyredner.set_use_gpu(False)

import numpy as np
# Restore deprecated numpy alias np.float to built-in float
np.float = float

import torch
# Save original torch.where for later patching
torch_where_orig = torch.where

def torch_where_patch(cond, x, y):
    """
    Patch torch.where so that Python float scalars are converted to
    tensors matching the dtype and device of the other branch.
    """
    # If x is float and y is tensor, convert x to tensor matching y
    if isinstance(x, float) and isinstance(y, torch.Tensor):
        x = torch.tensor(x, dtype=y.dtype, device=y.device)
    # If y is float and x is tensor, convert y to tensor matching x
    if isinstance(y, float) and isinstance(x, torch.Tensor):
        y = torch.tensor(y, dtype=x.dtype, device=x.device)
    # If both are tensors but dtypes differ, cast x to y.dtype
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.dtype != y.dtype:
        x = x.to(y.dtype)
    return torch_where_orig(cond, x, y)
# Override torch.where with patched version
torch.where = torch_where_patch

import time
import json
import csv
from bulletarm import env_factory

# ====== Test Configuration ======
TEST_CONFIG = {
    # List of object IDs (integers 1–84) to match JSON keys
    'OBJECT_LIST': list(range(1, 85)),
    # Maximum absolute value for initial x/y position
    'MAX_POS': 0.4,
    # List of noise standard deviations for adversarial perturbations
    'NOISE_LEVELS': [0.01, 0.05, 0.1],
    # Total number of tests to run
    'NUM_TESTS': 1_000_000,
    # Whether to enable rendering (True uses differentiable renderer)
    'RENDER': False,
    # Path to output CSV file
    'OUTPUT_CSV': 'grasp_adversarial_results.csv'
}


def sample_position(max_val):
    """
    Uniformly sample a 2D starting position in [-max_val, max_val]^2.
    """
    return np.random.uniform(-max_val, max_val, size=(2,))


def add_noise(vec, sigma):
    """
    Add zero-mean Gaussian noise with standard deviation sigma to vector vec.
    """
    return vec + np.random.normal(scale=sigma, size=vec.shape)


def init_env(object_id, render):
    """
    Initialize the grasping environment for a given object_id.

    object_id: integer ID of the object to grasp
    render: whether to enable differentiable rendering
    """
    cfg = {
        'render': render,
        'num_objects': 1,
        'object_index': object_id
    }
    if render:
        return env_factory.runCustomDemo(0, 'custom_object_grasping', cfg)
    else:
        return env_factory.createEnvs(0, 'object_grasping', cfg)


def run_adversarial_tests(config):
    """
    Main loop: run adversarial grasp tests using a single environment instance
    and log results to a CSV file.
    """
    # Randomly pick one object and noise level to reuse the same env
    obj_id = np.random.choice(config['OBJECT_LIST'])
    sigma = np.random.choice(config['NOISE_LEVELS'])

    # Initialize environment once
    env = init_env(obj_id, config['RENDER'])

    # Open CSV and write header
    with open(config['OUTPUT_CSV'], 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'test_id', 'object_id', 'noise_sigma',
            'init_x', 'init_y', 'final_reward', 'num_steps',
            'actions', 'heightmaps'
        ])
        writer.writeheader()

        # Execute tests
        for test_id in range(1, config['NUM_TESTS'] + 1):
            # Sample and perturb starting position
            clean_pos = sample_position(config['MAX_POS'])
            noisy_pos = add_noise(clean_pos, sigma)

            # Reset attack on the same environment
            env._resetAttack(noisy_pos)

            step_count = 0
            action_seq = []
            heightmaps = []
            reward = None
            done = False

            # Step until done
            while not done:
                # Get next planner action and add noise to parameters
                action = env.getNextAction()
                prim = action[0]
                params_noisy = add_noise(action[1:], sigma)
                action = np.concatenate(([prim], params_noisy))

                # Execute action and unpack observation tuple
                obs_tuple, reward, done = env.stepAttack(action)
                _, _, heightmap = obs_tuple

                step_count += 1
                action_seq.append(action.tolist())
                heightmaps.append(heightmap.tolist())

            # Write test result to CSV
            writer.writerow({
                'test_id': test_id,
                'object_id': obj_id,
                'noise_sigma': sigma,
                'init_x': clean_pos[0],
                'init_y': clean_pos[1],
                'final_reward': reward,
                'num_steps': step_count,
                'actions': json.dumps(action_seq),
                'heightmaps': json.dumps(heightmaps)
            })

            # Print progress every 10k tests
            if test_id % 10000 == 0:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Completed {test_id}/{config['NUM_TESTS']} tests on object {obj_id}")

    # Cleanup: close env and disconnect physics client
    env.close()
    import pybullet as pb
    pb.disconnect()


if __name__ == '__main__':
    run_adversarial_tests(TEST_CONFIG)

