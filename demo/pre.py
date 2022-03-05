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
#from torchinfo import summary 
from time import time 
from utils import logIoT, train, save_model, read_data, load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    num_classes = 2
    num_epochs = 60
    batch_size = 50
    learning_rate = 0.0001
    input_size = 30
    sequence_length = 50
    hidden_size = 128
    num_layers = 2
    seed = 42
    train_path = '../datasets/HDFS/log_train.csv'
    save_path = '../datasets/HDFS/model.h5'
    torch.manual_seed(seed)

    
    train_x, train_y = read_data(train_path, input_size, sequence_length)
    train_loader = load_data(train_x, train_y, batch_size)
    model = logIoT(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # Train the model
    print(f'Initially training model with learning rate = {learning_rate}')
    model = train(model, train_loader, learning_rate, num_epochs)
    save_model(model, save_path)


if __name__ == "__main__":

    main()

