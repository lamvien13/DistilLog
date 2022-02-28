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
import csv
from time import time 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
      super(LSTM, self).__init__()
      self.num_layers = num_layers
      self.hidden_size = hidden_size
      self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = 0.1, batch_first=True)
      self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
      out, _ = self.lstm(x)
      out = out[:, -1, :]
      out = self.fc(out)
      return out


def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path))
    return model

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def read_data(path, sequence_length):
    fi = pd.read_csv('../datasets/HDFS/vector.csv')
    vec = []
    vec = fi
    vec = np.array(vec)

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
    tensor_x = torch.Tensor(train_x) 
    tensor_y = torch.from_numpy(train_y)
    train_dataset = TensorDataset(tensor_x,tensor_y) 
    train_loader = DataLoader(train_dataset, batch_size = batch_size) 
    return train_loader



def train(model, train_loader, learning_rate, num_epochs):
    l1_regularization_strength = 0
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
            loss = criterion(output, target)
            total_loss += loss 

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
            loss.backward()          
            optimizer.step()

            if batch_idx % 10 == 0:
                done = (batch_idx+1) * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch+1}/{num_epochs} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {total_loss:.6f}')
        if total_loss < 65:
              break

    return model

