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

class CustomDataset(Dataset):
    def __init__(self, tensor_paths, device):
        self.tensor_paths = tensor_paths
        self.device = device
    
    def __len__(self):
        return len(self.tensor_paths)
    
    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]
        q_value_map, q2_output, action, reward = load_tensor(tensor_path, self.device)
        return q_value_map, q2_output, action, reward
    
class CustomModel(nn.Module):
    def __init__(self, input1_channels, input2_size, input3_size, hidden_size, output_size):
        super(CustomModel, self).__init__()

        # processing input 1
        self.conv1 = nn.Conv2d(input1_channels, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.maxpooling = nn.MaxPool2d(kernel_size=4)
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, hidden_size)

        # processing input 2
        self.fc3 = nn.Linear(input2_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        
        # processing input 3
        self.fc5 = nn.Linear(input3_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)

        # 输出层
        self.output_layer = nn.Linear(hidden_size * 3, output_size)

    def forward(self, input1, input2, input3):
        # Input1的处理
        x1 = torch.relu(self.conv1(input1))
        x1 = torch.relu(self.conv2(x1))
        x1 = torch.relu(self.conv3(x1))
        x1 = self.maxpooling(x1)
        x1 = torch.flatten(x1, 1)
        x1 = torch.relu(self.fc1(x1))
        x1 = torch.relu(self.fc2(x1))

        # Input2的处理
        x2 = torch.relu(self.fc3(input2))
        x2 = torch.relu(self.fc4(x2))

        # Input3的处理
        x3 = torch.relu(self.fc5(input3))
        x3 = torch.relu(self.fc6(x3))

        # 将三个输入连接起来
        x = torch.cat((x1, x2, x3), dim=1)

        # 输出层
        output = self.output_layer(x)

        return output

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
def generate_path_list():
    path_list = [f"dataset/{i}_{j}.pt" for i in range(86) for j in range(100)]
    return path_list

def load_tensor(tensor_path, device):
    tensor = torch.load(tensor_path, map_location=device).detach()
    q_value_map = tensor[:128*128].reshape(1,128,128)
    q2_output = tensor[128*128:-4]
    action = tensor[-4:-1]
    reward = tensor[-1:]
    return q_value_map, q2_output, action, reward

def split_dataset(train_data_ratio, eval_data_ratio, test_data_ratio, data_path_list, batch_size, device_name):
    train_size = int(train_data_ratio * len(data_path_list))
    eval_size = int(eval_data_ratio * len(data_path_list))
    test_size = len(data_path_list) - train_size - eval_size
    
    assert test_data_ratio + eval_data_ratio + train_data_ratio == 1
    
    train_dataset = CustomDataset(data_path_list[:train_size], device_name)
    eval_dataset = CustomDataset(data_path_list[train_size:train_size+eval_size], device_name)
    test_dataset = CustomDataset(data_path_list[train_size+eval_size:], device_name)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, eval_loader, test_loader

if __name__ == "__main__":
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_directory = './model_train_log'  
    os.makedirs(log_directory, exist_ok=True)  
    log_filename = os.path.join(log_directory, f'log_{timestamp}.log')
    
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    set_random_seed(42)
    if torch.cuda.is_available():
        device_name = torch.device('cuda')
    else:
        raise ValueError("cannot run on cpu device")
    
    data_path_list = generate_path_list()
    data_num = len(data_path_list)
    # print(data_path_list[:100])
    # assert 1 == 0
    # random.shuffle(data_path_list)

    train_data_ratio = 0.6
    eval_data_ratio = 0.2
    test_data_ratio = 0.2
    input1_channels = 1
    input2_size = 16
    input3_size = 3
    hidden_size = 64
    output_size = 1
    num_epochs = 20
    batch_size = 1
    log_train_loss = []
    log_eval_loss = []
    log_test_loss = []
    best_eval_loss = float('inf')

    MyModel = CustomModel(input1_channels, input2_size, input3_size, hidden_size, output_size)
    MyModel.cuda()
    print(MyModel)
    
    logging.info(f'Model info: {str(MyModel)}')
    
    logging.info("======================================")
    logging.info("Hyperparameters:")
    logging.info(f"Train Data Ratio: {train_data_ratio}")
    logging.info(f"Eval Data Ratio: {eval_data_ratio}")
    logging.info(f"Test Data Ratio: {test_data_ratio}")
    logging.info(f"Input1 Channels: {input1_channels}")
    logging.info(f"Input2 Size: {input2_size}")
    logging.info(f"Input3 Size: {input3_size}")
    logging.info(f"Hidden Size: {hidden_size}")
    logging.info(f"Output Size: {output_size}")
    logging.info(f"Number of Epochs: {num_epochs}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info("======================================")

    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(MyModel.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    test_loader, train_loader, eval_loader = split_dataset(test_data_ratio, 
                                                           train_data_ratio, 
                                                           eval_data_ratio, 
                                                           data_path_list, 
                                                           batch_size, 
                                                           device_name)
    
    for e in range(num_epochs):
        
        MyModel.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {e+1}/{num_epochs}, Training')
        for idx, data in enumerate(train_loader):
            q_value_map, q2_output, action, reward = data
            output = MyModel(q_value_map, q2_output, action)
            loss = criterion(output, reward)
            
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_bar.update()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        log_train_loss.append(avg_train_loss)
        
        MyModel.eval()
        eval_loss = 0
        eval_bar = tqdm(eval_loader, desc=f'Epoch {e+1}/{num_epochs}, Evaluating')
        with torch.no_grad():
            eval_acc = 0.0
            eval_cnt = 0.0
            wrong_pred_list = []
            for idx, data in enumerate(eval_loader):
                q_value_map, q2_output, action, reward = data
                output = MyModel(q_value_map, q2_output, action)
                loss = criterion(output, reward)
                eval_loss += loss.item()
                eval_bar.update()
                
                eval_cnt += 1
                if (output>0.5 and 1 == reward) or (output<=0.5 and 0 == reward):
                    eval_acc += 1
                else:
                    wrong_pred_list.append(eval_cnt)
            logging.info(wrong_pred_list)

        avg_eval_loss = eval_loss / len(eval_loader.dataset)
        log_eval_loss.append(avg_eval_loss)
        
        logging.info(f'Epoch {e+1}/{num_epochs}, Average Train Loss: {avg_train_loss:.4f}, Average Eval Loss: {avg_eval_loss:.4f}, eval_acc_rate: {eval_acc/eval_cnt:.4f}')
        print(f'Epoch {e+1}/{num_epochs}, Average Train Loss: {avg_train_loss:.4f}, Average Eval Loss: {avg_eval_loss:.4f}')
        print(f'Eval_acc_rate: {eval_acc/eval_cnt:.4f}')
        
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            torch.save(MyModel.state_dict(), f'./model_train_log/log_{timestamp}_ckpt.pth')
            
        
    MyModel.eval()
    test_loss = 0
    with torch.no_grad():
        test_cnt = 0.0
        test_acc = 0.0
        for idx, data in enumerate(test_loader):
            q_value_map, q2_output, action, reward = data
            output = MyModel(q_value_map, q2_output, action)
            loss = criterion(output, reward)
            log_test_loss.append(loss.detach().item())
            test_loss += loss.item()
            test_cnt += 1
            if (output>0.5 and 1 == reward) or (output<=0.5 and 0 == reward):
                test_acc += 1
            else:
                pass
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f'Average Test Loss: {avg_test_loss:.4f}')
    
    logging.info(f'Average Test Loss: {avg_test_loss:.4f}, Test_pred_acc: {test_acc/test_cnt:.4f}')
    
    plt.plot(log_train_loss, label='Train Loss')
    plt.plot(log_eval_loss, label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Evaluation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    
    plt.clf()
    
    plt.plot(log_test_loss, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Testing Loss')
    plt.savefig('test_loss_figure.png')
    
    plt.clf()
    
    print("=== Task Done === ")