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
from utils import LSTM, load_data, load_model 


batch_size = 100
input_size = 300
sequence_length = 50
hidden_size = 128
num_layers = 2
num_classes = 2 
test_split = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = '../datasets/HDFS/train.csv'
save_path = '../datasets/HDFS/model.h5'
test_path = '../datasets/HDFS/500log_test.csv'

fi = pd.read_csv('../datasets/HDFS/vector.csv')
vec = []
vec = fi
vec = np.array(vec)

test_logs_series = pd.read_csv(test_path)
test_logs_series = test_logs_series.values
test_total = len(test_logs_series)
test_sub = int(test_total/test_split)

def load_test(i):
    split = test_split
    sub = test_sub
    if i!=split-1:
        label = test_logs_series[i*sub:(i+1)*sub,1]
        logs_data = test_logs_series[i*sub:(i+1)*sub,0]
    else:
        label = test_logs_series[i*sub:,1]
        logs_data = test_logs_series[i*sub:,0]

    logs = []

    for i in tqdm (range(0,sub), desc = f"loading batch {i+1}/{split}: "):
        padding = np.full((sequence_length,300),-1)         
        data = logs_data[i]
        data = [int(n) for n in data.split()]
        if len(data) > sequence_length:
            data = data[-sequence_length:]
        for j in range(0,len(data)):
            padding[j] = vec[data[j]-1]
        padding = list(padding)
        logs.append(padding)
    logs = np.array(logs)
    train_x = logs
    train_y = label
    train_x = np.reshape(train_x, (train_x.shape[0], -1, 300))
    train_y = train_y.astype(int)
    return train_x, train_y



def test(model, criterion = nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        TP = 0 
        FP = 0
        FN = 0 
        TN = 0
        for i in range (0, test_split):        #################################################
            test_x, test_y = load_test(i)
            test_loader = load_data(test_x, test_y, batch_size)            
            for data, target in test_loader:
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

def main():      

    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    model = load_model(model, save_path)

    start_time = time()
    accuracy, test_loss, P, R, F1, TP, FP, TN, FN = test(model, criterion = nn.CrossEntropyLoss())
    test_loss /= (test_split*test_sub)

    print('false positive (FP): {}, false negative (FN): {}, true positive (TP): {}, true negative (TN): {}'.format(FP, FN, TP, TN))
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%). Total time = {time() - start_time}')
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(P, R, F1))

 
if __name__ == "__main__":

    main()

