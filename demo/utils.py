import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Parameter
from torch.nn.modules.module import Module
from tqdm import tqdm
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import math
from time import time 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = 0.5, batch_first=True)
        #fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 

        out, _ = self.lstm(x, (h0,c0))      
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path))
    return model

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def read_data(path, sequence_length):
    with open('../datasets/HDFS/hdfs_vector.json') as f:
        gdp_list = json.load(f)
        value = list(gdp_list.values())
        vec = []
        for i in range(0, len(value)): 
            vec.append(value[i])

    logs_series = pd.read_csv(path)
    logs_series = logs_series.values
    label = logs_series[:,1]
    logs_data = logs_series[:,0]
    logs = []
    for i in range(0,len(logs_data)):
        padding = np.full((sequence_length,300),-1)         
        data = logs_data[i]
        data = [int(n) for n in data.split()]
        if len(data) > sequence_length:
          data = data[-1*sequence_length:]
        for j in range(0,len(data)):
            padding[j] = vec[data[j]-1]
        padding = list(padding)
        logs.append(padding)
    logs = np.array(logs)
    train_x = logs
    train_y = np.array(label)
    train_x = np.reshape(train_x, (train_x.shape[0], -1, 300))
    train_y = train_y.astype(int)

    return train_x, train_y

def load_data(train_x, train_y, batch_size):
    tensor_x = torch.Tensor(train_x) # transform to torch tensor
    tensor_y = torch.from_numpy(train_y)
    train_dataset = TensorDataset(tensor_x,tensor_y) # create your dataset
    train_loader = DataLoader(train_dataset, batch_size = batch_size) # create your dataloader
    return train_loader



def train(model, train_loader, learning_rate, num_epochs):
    l1_regularization_strength = 0
    l2_regularization_strength = 0.00001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.0001)  
    model.train()
    
    for epoch in range(num_epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        total_loss = 0
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            #print(output, target)
            loss = criterion(output, target)
            total_loss += loss 
            #loss = F.nll_loss(output, target)

            l1_reg = torch.tensor(0.).to(device)
            for module in model.modules():
              mask = None
              weight = None
              for name, buffer in module.named_buffers():
                if 'mask' in name:
                  mask = buffer
              for name, param in module.named_parameters():
                if 'orig' in name:
                  weight = param
              if mask is not None and weight is not None:
                l1_reg += torch.norm(mask*weight, 1)
            
            total_loss += l1_regularization_strength * l1_reg
            #loss = F.nll_loss(output, target)
            loss.backward()          
            optimizer.step()

            if batch_idx % 10 == 0:
                done = (batch_idx+1) * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch+1} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {total_loss:.6f}')
    return model

