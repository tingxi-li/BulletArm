import os
import sys
import time
import copy
import math
import collections
from tqdm import tqdm
import datetime
import threading

import torch

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
sys.path.append('./')
sys.path.append('..')
from bulletarm_baselines.fc_dqn.scripts.create_agent import createAgent
from bulletarm_baselines.fc_dqn.storage.buffer import QLearningBufferExpert, QLearningBuffer
from bulletarm_baselines.logger.logger import Logger
from bulletarm_baselines.logger.baseline_logger import BaselineLogger
from bulletarm_baselines.fc_dqn.utils.schedules import LinearSchedule
from bulletarm_baselines.fc_dqn.utils.env_wrapper import EnvWrapper

from bulletarm_baselines.fc_dqn.utils.parameters import *
from bulletarm_baselines.fc_dqn.utils.torch_utils import augmentBuffer, augmentBufferD4
from bulletarm_baselines.fc_dqn.scripts.fill_buffer_deconstruct import fillDeconstructUsingRunner


ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluation():
    import pdb;    
    eval_thread = None
    start_time = time.time()
    if seed is not None:
        set_seed(seed)
    # setup env
    num_eval_processes = 0
    eval_envs = EnvWrapper(num_eval_processes, env, env_config, planner_config)

    # setup agent
    eval_agent = createAgent(test=True)
    # pdb.set_trace()
    if load_model_pre:
        eval_agent.loadModel(load_model_pre)

    base_dir = os.path.join(log_pre, '{}_{}_{}'.format(alg, model, env))
    if note:
        base_dir += '_'
        base_dir += note
    if not log_sub:
        timestamp = time.time()
        timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d.%H:%M:%S')
        log_dir = os.path.join(base_dir, timestamp)
    else:
        log_dir = os.path.join(base_dir, log_sub)

    # logger = Logger(log_dir, env, 'train', num_processes, max_episode, log_sub)
    
    hyper_parameters['model_shape'] = eval_agent.getModelStr()
    logger = BaselineLogger(log_dir, checkpoint_interval=save_freq, num_eval_eps=num_eval_episodes, hyperparameters=hyper_parameters, eval_freq=eval_freq)
    logger.saveParameters(hyper_parameters)

    eval_envs.envs.setObjectInitMetaData()
    states, in_hands, obs = eval_envs.reset()
    
    evaled = 0
    sum_of_reward = 0
    temp_reward = [[] for _ in range(1)]
    if not no_bar:
        eval_bar = tqdm(total=num_eval_episodes)
        
    while evaled < num_eval_episodes:
        q_value_maps, actions_star_idx, actions_star = eval_agent.getEGreedyActions(states, in_hands, obs, 0)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        states_, in_hands_, obs_, rewards, dones = eval_envs.step(actions_star, auto_reset=True)
        rewards = rewards.numpy()
        dones = dones.numpy()
        states = copy.copy(states_)
        in_hands = copy.copy(in_hands_)
        obs = copy.copy(obs_)
        for i, r in enumerate(rewards.reshape(-1)):
            temp_reward[i].append(r)
        evaled += int(np.sum(dones))
        for i, d in enumerate(dones.astype(bool)):
            if d:
                R = 0
                for r in reversed(temp_reward[i]):
                    R = r + gamma * R
                # eval_rewards.append(R)
                temp_reward[i] = []
                sum_of_reward += R
        if not no_bar:
            eval_bar.update(evaled - eval_bar.n)

    if not no_bar:
        eval_bar.close()
        
    print("avg evaled reward: ", sum_of_reward/evaled)
        
        
if __name__ == "__main__":
    evaluation()