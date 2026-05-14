#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# Ensure we run from the repo root so assets are found
tool_dir   = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.dirname(tool_dir)
os.chdir(repo_root)

import numpy as np
# Restore deprecated alias
np.float = float

import time
import csv
from bulletarm import env_factory

# ====== Test Configuration ======
TEST_CONFIG = {
    'ENV_NAME':     'object_grasping',       # realistic grasping task
    'OBJECT_LIST':  list(range(1, 85)),      # object IDs 1–84
    'MAX_POS':      0.4,                     # workspace bounds for initial XY
    'NOISE_LEVELS': [0.01, 0.05, 0.1],       # σ for adversarial noise
    'NUM_TESTS':    1000,                    # total trials
    'RENDER':       False,
    'OUTPUT_CSV':   'grasp_adversarial_results.csv'
}


def add_noise(vec, sigma):
    """Add zero-mean Gaussian noise with std sigma to numpy vector."""
    return vec + np.random.normal(scale=sigma, size=vec.shape)


def sample_position(max_val):
    """Uniformly sample a 2D non-adversarial starting position"""
    return np.random.uniform(-max_val, max_val, size=(2,))


def init_env(env_name, render, object_id):
    """
    Instantiate a runner (num_envs=1) for the specified env and object.
    runner.reset() yields (obs_batch, reward_batch, done_batch).
    """
    cfg = {'render': render, 'num_objects': 1, 'object_index': object_id}
    return env_factory.createEnvs(1, env_name, cfg)


def run_adversarial_tests(config):
    """Run adversarial tests and write results to CSV."""
    with open(config['OUTPUT_CSV'], 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'test_id','object_id','noise_sigma',
            'init_x','init_y','final_reward','num_steps'
        ])
        writer.writeheader()

        for test_id in range(1, config['NUM_TESTS'] + 1):
            obj_id = np.random.choice(config['OBJECT_LIST'])
            sigma  = np.random.choice(config['NOISE_LEVELS'])

            # Sample clean init, but we cannot use _resetAttack, so object placement random
            clean_pos = sample_position(config['MAX_POS'])

            # Create runner for this object
            env = init_env(config['ENV_NAME'], config['RENDER'], obj_id)

            # Reset batch: returns tuple(obs_batch, reward_batch, done_batch)
            obs_batch, reward_batch, done_batch = env.reset()
                        # Extract scalars for the single env from batch arrays
            reward = float(np.squeeze(reward_batch))
            done   = bool(np.squeeze(done_batch))
            step_count   = 0
            final_reward = reward

            # Adversarial loop: perturb each action
            while not done:
                action_batch = env.getNextAction()
                # single-env action
                action = action_batch[0]
                prim   = action[0]
                params = add_noise(action[1:], sigma)
                action = np.concatenate(([prim], params))

                # step returns same batch-structure
                obs_batch, reward_batch, done_batch = env.step(action)
                reward = float(reward_batch[0])
                done   = bool(done_batch[0])
                final_reward = reward
                step_count  += 1

            # Clean up
            env.close()
            import pybullet as pb
            pb.disconnect()

            writer.writerow({
                'test_id':      test_id,
                'object_id':    obj_id,
                'noise_sigma':  sigma,
                'init_x':       clean_pos[0],
                'init_y':       clean_pos[1],
                'final_reward': final_reward,
                'num_steps':    step_count
            })

            if test_id % 100 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Completed {test_id}/{config['NUM_TESTS']}")

if __name__ == '__main__':
    # run_adversarial_tests(TEST_CONFIG)
    cfg = {'render': False, 'num_objects': 1, 'object_index': 0}
    env1 = env_factory.createEnvs(0, "object_grasping", cfg)
    _metadata = env1.setObjectInitMetaData()
    env1.reset()
    act1 = env1.getNextAction()
    _, re1, done1 = env1.step(act1)
    env1.close()
    
    env2 = env_factory.createEnvs(0, "custom_object_grasping", cfg)
    _ = env2.setObjectInitMetaData(_metadata)
    env2.reset()
    act2 = env2.getNextAction()
    _, re2, done2 = env2.step(act2)
    env2.close()
    
    print(re1, re2)
    # import pdb; pdb.set_trace()
    
    
    
    