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
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix
import math
from time import time 
batch_size = 64 
sequence_length = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_data(test_path, sequence_length):
    logs_series = pd.read_csv(test_path)
    logs_series = logs_series.values
    total = len(logs_series)
    label = logs_series[:,1]
    logs_data = logs_series[:,0]
    logs = []

    for i in range(0, total):
        data = logs_data[i]
        data = [int(n) for n in data.split()]
        if len(data) > sequence_length:
            data = data[-sequence_length:]
        while len(data) < sequence_length:
            data.append(0)
        logs.append(data)

    logs = np.array(logs)
    x = logs
    y = label.astype(int)

    return x, y

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embbed = nn.Embedding(num_embeddings = sequence_length, embedding_dim = input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = 0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 

        out = self.embbed(x)
        out, _ = self.lstm(out, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path))
    return model

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_data(train_x, train_y, batch_size):
    tensor_x = torch.Tensor(train_x) # transform to torch tensor
    tensor_x = tensor_x.type(torch.LongTensor)
    tensor_y = torch.from_numpy(train_y)
    train_dataset = TensorDataset(tensor_x,tensor_y) # create your dataset
    train_loader = DataLoader(train_dataset, batch_size = batch_size, drop_last = True) # create your dataloader
    return train_loader

def train(model, train_loader, learning_rate, num_epochs):
    l1_regularization_strength = 0.00001
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
        if epoch > 10 and total_loss < 5:
            break
    return model


def apply_weight_sharing(model, bits=5):
    for module in model.children():
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num=2**bits)
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(mat.data.reshape(-1,1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        mat.data = new_weight
        module.weight.data = torch.from_numpy(mat.toarray()).to(dev)
    return model


def test(model, test_loader, criterion = nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        TP = 0 
        FP = 0
        FN = 0 
        TN = 0
   #################################################         
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target) # sum up batch loss
            
            output = torch.sigmoid(output)[:, 0].cpu().detach().numpy()
            predicted = (output < 0.2).astype(int)
            target = np.array([y.cpu() for y in target])
            #print(predicted, label)
            TP += ((predicted == 1) * (target == 1)).sum()
            FP += ((predicted == 1) * (target == 0)).sum()
            FN += ((predicted == 0) * (target == 1)).sum()
            TN += ((predicted == 0) * (target == 0)).sum()
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)   
        accuracy = 100 * (TP + TN)/(TP + TN + FP + FN)        
    return accuracy, test_loss, P, R, F1, TP, FP, TN, FN