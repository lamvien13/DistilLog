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
from utils import DistilLog, load_data, load_model 


batch_size = 50
input_size = 30
sequence_length = 50
hidden_size = 128
num_layers = 2
num_classes = 2 
split = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = '../datasets/HDFS/train.csv'
save_original_path = '../datasets/HDFS/model/teacher.pth'
save_pruned_path = '../datasets/HDFS/model/pruned_model.pth'
test_path = '../datasets/HDFS/test.csv'

fi = pd.read_csv('../datasets/HDFS/pca_vector.csv', header = None)
vec = []
vec = fi
vec = np.array(vec)

test_logs_series = pd.read_csv(test_path)
test_logs_series = test_logs_series.values
test_total = len(test_logs_series)
sub = int(test_total/split)

def mod(l, n):
    """ Truncate or pad a list """
    r = l[-1*n:]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r

def load_test(i):
    if i!=split-1:
        label = test_logs_series[i*sub:(i+1)*sub,1]
        logs_data = test_logs_series[i*sub:(i+1)*sub,0]
    else:
        label = test_logs_series[i*sub:,1]
        logs_data = test_logs_series[i*sub:,0]
    logs = []

    for logid in range(0,len(logs_data)):
        ori_seq = [
            int(eventid) for eventid in logs_data[logid].split()]
        seq_pattern = mod(ori_seq, sequence_length)
        vec_pattern = []

        for event in seq_pattern:
            if event == 0:
                vec_pattern.append([-1]*input_size)
            else:
                vec_pattern.append(vec[event-1])  
        logs.append(vec_pattern)
    logs = np.array(logs)
    train_x = logs
    train_y = label
    train_x = np.reshape(train_x, (train_x.shape[0], -1, input_size))
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
        for i in range (0, split):        #################################################
            test_x, test_y = load_test(i)
            test_loader = load_data(test_x, test_y, batch_size)            
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                test_loss += criterion(output, target) # sum up batch loss
                
                output = torch.sigmoid(output)[:, 0].cpu().detach().numpy()
                predicted = (output < 0.2).astype(int)
                target = np.array([y.cpu() for y in target])

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

    original_model = DistilLog(input_size, hidden_size, num_layers, num_classes, is_bidirectional = False).to(device)
    pruned_model = DistilLog(input_size, hidden_size, num_layers, num_classes, is_bidirectional = False).to(device)
    original_model = load_model(original_model, save_original_path)
    pruned_model = load_model(pruned_model, save_pruned_path)


    start_time = time()
    accuracy, test_loss, P, R, F1, TP, FP, TN, FN = test(original_model, criterion = nn.CrossEntropyLoss())
    test_loss /= (split*sub)

    print('false positive (FP): {}, false negative (FN): {}, true positive (TP): {}, true negative (TN): {}'.format(FP, FN, TP, TN))
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%). Total time = {time() - start_time}')
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(P, R, F1))

    start_time = time()
    accuracy, test_loss, P, R, F1, TP, FP, TN, FN = test(pruned_model, criterion = nn.CrossEntropyLoss())
    test_loss /= (split*sub)

    print('false positive (FP): {}, false negative (FN): {}, true positive (TP): {}, true negative (TN): {}'.format(FP, FN, TP, TN))
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%). Total time = {time() - start_time}')
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(P, R, F1))

 
if __name__ == "__main__":

    main()

