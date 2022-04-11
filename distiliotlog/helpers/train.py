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
import math
import csv
from time import time 
from torchinfo import summary
from utils import save_model, train, read_data, load_data
from utils import DistilLog
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 2
batch_size = 50
learning_rate = 0.001
hidden_size = 128
input_size = 30
sequence_length = 50
num_layers = 2

train_path = '../datasets/BGL/new_train.csv'
save_teacher_path = '../datasets/BGL/model/teacher_model.pth'
save_student_path = '../datasets/BGL/model/student_model.pth'

Teacher = DistilLog(input_size, hidden_size, num_layers, num_classes).to(device)
#summary(Teacher, input_size=(50, 50, 30))

train_x, train_y = read_data(train_path, input_size, sequence_length)
train_loader = load_data(train_x, train_y, batch_size)

Teacher = train(Teacher, train_loader, learning_rate, num_epochs = 200)
save_model(Teacher, save_teacher_path)