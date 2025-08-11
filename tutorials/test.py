#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# Suppress GTK cursor warnings when PyBullet touches DISPLAY
os.environ["DISPLAY"] = ""

import time
import csv
import numpy as np
# restore deprecated alias
np.float = float

from bulletarm import env_factory

# ====== Test Configuration ======
TEST_CONFIG = {
    'ENV_NAME':     'object_grasping',       # realistic grasping task
    'OBJECT_LIST':  list(range(1, 85)),      # object IDs 1–84
    'MAX_POS':      0.4,                     # workspace bounds for initial XY
    'NOISE_LEVELS': [0.01, 0.05, 0.1],       # σ for adversarial noise
    'NUM_TESTS':    70000,                    # total trials
    'RENDER':       False,
    'OUTPUT_CSV':   'long_grasp_adversarial_results.csv'
}

def add_noise(vec, sigma):
    """Add zero-mean Gaussian noise with std sigma to a numpy vector."""
    return vec + np.random.normal(scale=sigma, size=vec.shape)

def sample_position(max_val):
    """Uniformly sample a 2D starting XY in [-max_val, max_val]."""
    return np.random.uniform(-max_val, max_val, size=(2,))

def init_env(env_name, render, object_id):
    """
    Instantiate a SingleRunner (num_processes=0), so getNextAction()
    returns a 1-D action vector and step(action) accepts a 1-D vector.
    """
    cfg = {'render': render, 'num_objects': 1, 'object_index': object_id}
    # 0 → SingleRunner
    return env_factory.createEnvs(0, env_name, cfg)

def extract_scalar(x):
    """
    Given x that might be a float, numpy scalar, or small numpy array,
    return the first element as a Python float/bool.
    """
    arr = np.atleast_1d(x)      # wrap scalar→1d array, array stays array
    return arr.flatten()[0]

def run_adversarial_tests(cfg):
    with open(cfg['OUTPUT_CSV'], 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'test_id','object_id','noise_sigma',
            'init_x','init_y','final_reward','num_steps'
        ])
        writer.writeheader()

        for test_id in range(1, cfg['NUM_TESTS'] + 1):
            # 1) pick random object & noise
            obj_id = np.random.choice(cfg['OBJECT_LIST'])
            sigma  = np.random.choice(cfg['NOISE_LEVELS'])
            clean_pos = sample_position(cfg['MAX_POS'])

            # 2) init SingleRunner
            env = init_env(cfg['ENV_NAME'], cfg['RENDER'], obj_id)

            # 3) reset observation only
            _ = env.reset()

            done = False
            step_count = 0
            final_reward = 0.0

            # 4) adversarial loop
            while not done:
                # a) get 1-D action vector
                action = env.getNextAction()
                prim   = action[0]
                params = add_noise(action[1:], sigma)
                action = np.concatenate(([prim], params))

                # b) step; this returns (_, rewards, dones)
                _, reward_batch, done_batch = env.step(action)

                # c) extract true Python scalar
                reward = float(extract_scalar(reward_batch))
                done   = bool(extract_scalar(done_batch))

                final_reward = reward
                step_count  += 1

            # 5) cleanup
            env.close()
            import pybullet as pb
            pb.disconnect()

            # 6) log result
            writer.writerow({
                'test_id':      test_id,
                'object_id':    obj_id,
                'noise_sigma':  sigma,
                'init_x':       clean_pos[0],
                'init_y':       clean_pos[1],
                'final_reward': final_reward,
                'num_steps':    step_count
            })

            # 7) progress indicator
            if test_id % 100 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Completed {test_id}/{cfg['NUM_TESTS']} tests")

if __name__ == '__main__':
    run_adversarial_tests(TEST_CONFIG)
