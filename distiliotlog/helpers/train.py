import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from distiliotlog.models.gru import GRU

#parameters
num_classes = 2
num_epochs = 10
batch_size = 100
learning_rate = 0.1
input_size = 20
sequence_length = 300
hidden_size = 128
num_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    Model = GRU(input_size, hidden_size, num_layers, num_classes)
    