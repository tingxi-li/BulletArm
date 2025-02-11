import torch
import re
import numpy as np
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

import logging
import time

from train import CustomDataset
from train import CustomModel
from train import set_random_seed
from train import generate_path_list
from train import split_dataset

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device_name = torch.device('cuda')
    else:
        raise ValueError("cannot run on cpu device")
    
    set_random_seed(42)
    data_path_list = generate_path_list()
    data_num = len(data_path_list)
    # random.shuffle(data_path_list)
    
    train_data_ratio = 0.8
    eval_data_ratio = 0.1
    test_data_ratio = 0.1
    input1_channels = 1
    input2_size = 16
    input3_size = 3
    hidden_size = 64
    output_size = 1
    num_epochs = 100
    batch_size = 1
    log_train_loss = []
    log_eval_loss = []
    log_test_loss = []
    criterion = nn.BCEWithLogitsLoss()
    MyModel = CustomModel(input1_channels, input2_size, input3_size, hidden_size, output_size)
    MyModel.eval()
    MyModel.cuda()
    
    train_loader, eval_loader, test_loader = split_dataset(train_data_ratio, 
                                                           eval_data_ratio, 
                                                           test_data_ratio, 
                                                           data_path_list, 
                                                           batch_size, 
                                                           device_name)
    
    checkpoint = torch.load('model_train_log/log_20240912-133602_ckpt.pth')
    MyModel.load_state_dict(checkpoint)
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        for idx, data in enumerate(test_loader):
            q_value_map, q2_output, action, reward = data
            output = MyModel(q_value_map, q2_output, action)
            sigmoid_output = torch.sigmoid(output)
            loss = criterion(output, reward)
            log_test_loss.append(loss.detach().item())
            test_loss += loss.item()

            # prediction and ground truth
            predicted = (sigmoid_output > 0.5).float()
            total += reward.size(0)
            correct += (predicted == reward).sum().item()
            true_positives += ((predicted == 1) & (reward == 1)).sum().item()
            false_positives += ((predicted == 1) & (reward == 0)).sum().item()
            false_negatives += ((predicted == 0) & (reward == 1)).sum().item()
            true_negatives += ((predicted == 0) & (reward == 0)).sum().item()

    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(false_positives, true_negatives)

    
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f'Average Test Loss: {avg_test_loss:.4f}')
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_directory = './model_inference_log'  
    os.makedirs(log_directory, exist_ok=True)  
    log_filename = os.path.join(log_directory, f'log_{timestamp}.log')
        
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'Precision: {precision:.4f}')
    logging.info(f'Recall: {recall:.4f}')
    logging.info(f'F1 Score: {f1:.4f}')
    logging.info(f'False Negatives: {false_negatives}, True Positives: {true_positives}')
    logging.info(f'False Positives: {false_positives}, True Negatives: {true_negatives}')
    
    logging.info(f'Average Test Loss: {avg_test_loss:.4f}')