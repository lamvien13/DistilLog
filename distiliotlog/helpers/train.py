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
import torch.nn.functional as F
import math
from time import time 

num_classes = 2
num_epochs = 40
batch_size = 100
learning_rate = 0.001
input_size = 300
sequence_length = 50
hidden_size = 128
num_layers = 2
seed = 42
split = 50
log_after = 10 # How many batches to wait before logging training status

torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./data/hdfs_vector.json') as f:
    gdp_list = json.load(f)
    value = list(gdp_list.values())
    vec = []
    for i in range(0, len(value)):
        vec.append(value[i])


def read_data(path):
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

train_path = './data/log_train.csv'
save_path = './model.pth'

def load_data(train_x, train_y):
    tensor_x = torch.Tensor(train_x) # transform to torch tensor
    tensor_y = torch.from_numpy(train_y)
    train_dataset = TensorDataset(tensor_x,tensor_y) # create your dataset
    train_loader = DataLoader(train_dataset, batch_size = batch_size) # create your dataloader
    return train_loader

def train(model):
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
            loss.backward()

            optimizer.step()
            if batch_idx % log_after == 0:
                done = (batch_idx+1) * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch+1} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {total_loss:.6f}')
    return model

def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')

def prune(model, pruning_percentage):
    all_weights = []
    pruning_percentage = pruning_percentage
    for p in LSTM.parameters(model):
        if len(p.data.size()) == 2:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_percentage)

    pruned_inds_by_layer = []
    w_original = []
    w_pruned = []
    print(threshold)

    for p in LSTM.parameters(model):
        pruned_inds = 'None'
        if len(p.data.size()) == 2:
            pruned_inds = p.data.abs() < threshold
            w_original.append(p.cpu().clone())
            p.data[pruned_inds] = 0.
            w_pruned.append(p.cpu().clone())
        pruned_inds_by_layer.append(pruned_inds)

model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.0001)  
initial_optimizer_state_dict = optimizer.state_dict()

# Train the model
model = train(model)
torch.save(model, save_path)

#prune and re-train the model
for i in range (1, 9):
    prune(model, 10*i)
    optimizer.load_state_dict(initial_optimizer_state_dict)
    model = train(model)

print_nonzeros(model)