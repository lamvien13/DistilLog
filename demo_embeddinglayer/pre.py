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
from utils import LSTM, train, prepare_data, load_data, save_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
CUDA_LAUNCH_BLOCKING=1
num_classes = 2
num_epochs = 500
batch_size = 64
learning_rate = 0.01
input_size = 50
sequence_length = 50
hidden_size = 128
num_layers = 2
seed = 42
train_path = '../datasets/HDFS/log_train.csv'
save_path = '../datasets/HDFS/model.pth'
torch.manual_seed(seed)




def main():
    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
  
    # Train the model
    print(f'Initially training model with learning rate = {learning_rate}')          
    x, y = prepare_data(train_path, sequence_length)
    train_loader = load_data(x, y, batch_size=batch_size)
    model = train(model, train_loader, learning_rate, num_epochs)
    save_model(model, save_path)


if __name__ == "__main__":
    main()