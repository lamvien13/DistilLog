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
from utils import LSTM, read_data, load_data, train, save_model, load_model 

num_classes = 2
num_epochs = 5
batch_size = 100
learning_rate = 0.01
input_size = 300
sequence_length = 50
hidden_size = 128
num_layers = 2
split = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = '../datasets/HDFS/log_train.csv'
save_path = '../datasets/HDFS/model.h5'
test_path = '../datasets/HDFS/500log_test.csv'

with open('../datasets/HDFS/hdfs_vector.json') as f:
    gdp_list = json.load(f)
    value = list(gdp_list.values())
    vec = []
    for i in range(0, len(value)): 
        vec.append(value[i])

logs_series = pd.read_csv(test_path)
logs_series = logs_series.values
total = len(logs_series)
sub = int(total/split)

def load_test(i):
    if i==split-1:
        label = logs_series[i*sub:,1]
        logs_data = logs_series[i*sub:,0]
    else:
        label = logs_series[i*sub:(i+1)*sub,1]
        logs_data = logs_series[i*sub:(i+1)*sub,0]
    logs = []

    for i in tqdm (range(0,sub), desc = f"loading batch: {sub+1}"):
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



def iterative_pruning_finetuning(model, train_loader, learning_rate, num_epochs, save_path,
                                 lstm_prune_amount,
                                 linear_prune_amount,
                                 num_iter):

    num_iterations = num_iter 
    for i in range(0, num_iterations):
        print("Pruning and Finetuning {}/{}".format(i + 1, num_iterations))
        print("Pruning...")


        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LSTM):
                prune.l1_unstructured(module, 
                                    name="weight_ih_l0",
                                    amount=lstm_prune_amount)
                prune.l1_unstructured(module,
                                    name="weight_hh_l0",
                                    amount=lstm_prune_amount)
                prune.l1_unstructured(module,
                                    name="weight_ih_l1",
                                    amount=lstm_prune_amount)
                prune.l1_unstructured(module,
                                    name="weight_hh_l1",
                                    amount=lstm_prune_amount)
                prune.l1_unstructured(module, 
                                    name="bias_ih_l0",
                                    amount=lstm_prune_amount)
                prune.l1_unstructured(module,
                                    name="bias_hh_l0",
                                    amount=lstm_prune_amount)
                prune.l1_unstructured(module,
                                    name="bias_ih_l1",
                                    amount=lstm_prune_amount)
                prune.l1_unstructured(module,
                                    name="bias_hh_l1",
                                    amount=lstm_prune_amount)
            elif isinstance(module, torch.nn.Linear):    
                prune.l1_unstructured(module,
                                      name="weight",
                                      amount=linear_prune_amount)
                prune.l1_unstructured(module,
                                      name="bias",
                                      amount=linear_prune_amount)
                
                                                                                                                                                         
        print("Fine-tuning...")
        model = train(model, train_loader, learning_rate, num_epochs)

        save_model(model, save_path)
        model = load_model(model, save_path)

    return model

def test(model, criterion = nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        TP = 0 
        FP = 0
        FN = 0 
        TN = 0
        for i in range (0, split):
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



def remove_parameters(model):
#prune.remove(module, 'weight')
  for name, module in model.named_modules():
      # prune 20% of connections in all 2D-conv layers
      if isinstance(module, torch.nn.LSTM):
          prune.remove(module, 'weight_ih_l0')
          prune.remove(module, 'weight_hh_l0')
          prune.remove(module, 'weight_ih_l1')
          prune.remove(module, 'weight_hh_l1')
          prune.remove(module, 'bias_ih_l0')
          prune.remove(module, 'bias_hh_l0')
          prune.remove(module, 'bias_ih_l1')
          prune.remove(module, 'bias_hh_l1')
      # prune 40% of connections in all linear layers
      elif isinstance(module, torch.nn.Linear):
          prune.remove(module, 'weight')
          prune.remove(module, 'bias')
  print_nonzeros(model)

  return model



def main():
    
    
    ### LOAD TEST DATA ###
       

    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    model = load_model(model, save_path)

    train_x, train_y = read_data(train_path, sequence_length)
    train_loader = load_data(train_x, train_y, batch_size)

    print("Iterative Pruning + Fine-Tuning...")
    pruned_model = copy.deepcopy(model)

    iterative_pruning_finetuning(pruned_model, train_loader, learning_rate, num_epochs, save_path,
                                 lstm_prune_amount = 0.03,
                                 linear_prune_amount = 0.03,
                                 num_iter = 5)
    remove_parameters(pruned_model)

    print("Finished Prunning")

    ###TEST THE MODEL AFTER PRUNNING

    start_time = time()
    accuracy, test_loss, P, R, F1, TP, FP, TN, FN = test(pruned_model, criterion = nn.CrossEntropyLoss())
    test_loss /= (split*sub)


    print('false positive (FP): {}, false negative (FN): {}, true positive (TP): {}, true negative (TN): {}'.format(FP, FN, TP, TN))
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%). Total time = {time() - start_time}')
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(P, R, F1))
    save_model(pruned_model, save_path)
    print_nonzeros(pruned_model)


if __name__ == "__main__":

    main()

