import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from tqdm import tqdm
import math
import copy
from time import time 
from utils import LSTM, test, prepare_data, load_data, load_model 

num_classes = 2
batch_size = 64
learning_rate = 0.01
input_size = 50
sequence_length = 50
hidden_size = 128
num_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = '../datasets/HDFS/log_train.csv'
save_path = '../datasets/HDFS/model.pth'
save_pruned_path = '../datasets/HDFS/pruned_model.pth'
test_path = '../datasets/HDFS/500log_test.csv'



def main():      
    x, y = prepare_data(test_path, sequence_length)
    test_loader = load_data(x, y, batch_size=batch_size)

    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    model = load_model(model, save_path)

    start_time = time()
    accuracy, test_loss, P, R, F1, TP, FP, TN, FN = test(model, test_loader, criterion = nn.CrossEntropyLoss())
    test_loss /= (len(test_loader))

    print("initial model:")
    print('false positive (FP): {}, false negative (FN): {}, true positive (TP): {}, true negative (TN): {}'.format(FP, FN, TP, TN))
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%). Total time = {time() - start_time}')
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(P, R, F1))

    
    print("pruned model:")
    pruned_model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    pruned_model = load_model(pruned_model, save_path)

    start_time = time()
    accuracy, test_loss, P, R, F1, TP, FP, TN, FN = test(pruned_model, test_loader, criterion = nn.CrossEntropyLoss())
    test_loss /= (len(test_loader))

    print("initial model:")
    print('false positive (FP): {}, false negative (FN): {}, true positive (TP): {}, true negative (TN): {}'.format(FP, FN, TP, TN))
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%). Total time = {time() - start_time}')
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(P, R, F1))
 
if __name__ == "__main__":

    main()

