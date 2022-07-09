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
#from torchinfo import summary
import math
import copy
from time import time 
from utils import DistilLog, read_data, load_data, train, save_model, load_model 
from test import test

num_classes = 2
num_epochs = 40
batch_size = 50
learning_rate = 0.0005
input_size = 30
sequence_length = 50
hidden_size = 128
num_layers = 2
split = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = '../datasets/HDFS/train.csv'
save_original_path = '../datasets/HDFS/model/teacher.pth'
save_pruned_path = '../datasets/HDFS/model/pruned_model.pth'
test_path = '../datasets/HDFS/test.csv'

fi = pd.read_csv('../datasets/HDFS/pca_vector.csv')
vec = []
vec = fi
vec = np.array(vec)

logs_series = pd.read_csv(test_path)
logs_series = logs_series.values
total = len(logs_series)
sub = int(total/split)  

def iterative_pruning_finetuning(model, train_loader, learning_rate, num_epochs, save_path,
                                 lstm_prune_amount,
                                 linear_prune_amount,
                                 num_iterations):

    for i in range(0, num_iterations):
        print("Pruning and Finetuning {}/{}".format(i + 1, num_iterations))
        print("Pruning...")


        for name, module in model.named_modules():
            if isinstance(module, torch.nn.GRU):
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
        
        if (i+1)%3 == 0:
            start_time = time()
            accuracy, test_loss, P, R, F1, TP, FP, TN, FN = test(model, criterion = nn.CrossEntropyLoss())
            test_loss /= (split*sub)


            print('false positive (FP): {}, false negative (FN): {}, true positive (TP): {}, true negative (TN): {}'.format(FP, FN, TP, TN))
            print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%). Total time = {time() - start_time}')
            print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                        .format(P, R, F1))
            print_nonzeros(model)
        

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



def remove_parameters(model):
  for name, module in model.named_modules():
      if isinstance(module, torch.nn.GRU):
          prune.remove(module, 'weight_ih_l0')
          prune.remove(module, 'weight_hh_l0')
          prune.remove(module, 'weight_ih_l1')
          prune.remove(module, 'weight_hh_l1')
          prune.remove(module, 'bias_ih_l0')
          prune.remove(module, 'bias_hh_l0')
          prune.remove(module, 'bias_ih_l1')
          prune.remove(module, 'bias_hh_l1')
      elif isinstance(module, torch.nn.Linear):
          prune.remove(module, 'weight')
          prune.remove(module, 'bias')
  print_nonzeros(model)

  return model



def main():

    train_x, train_y = read_data(train_path, input_size, sequence_length)
    train_loader = load_data(train_x, train_y, batch_size)

    model = DistilLog(input_size, hidden_size, num_layers, num_classes, is_bidirectional = False).to(device)
    model = load_model(model, save_original_path)
    #summary(model, input_size=(50, 50, 20))

    pruned_model = copy.deepcopy(model)
    print("Iterative Pruning + Fine-Tuning...")
    
    iterative_pruning_finetuning(pruned_model, train_loader, learning_rate, num_epochs, save_pruned_path,
                                 lstm_prune_amount = 0.15,
                                 linear_prune_amount = 0.15,
                                 num_iterations = 30)
    remove_parameters(pruned_model)

    print("Finished Prunning")

    ###TEST THE MODEL AFTER PRUNNING '''

    start_time = time()
    accuracy, test_loss, P, R, F1, TP, FP, TN, FN = test(pruned_model, criterion = nn.CrossEntropyLoss())
    test_loss /= (split*sub)


    print('false positive (FP): {}, false negative (FN): {}, true positive (TP): {}, true negative (TN): {}'.format(FP, FN, TP, TN))
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%). Total time = {time() - start_time}')
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(P, R, F1))
    save_model(pruned_model, save_pruned_path)
    print_nonzeros(pruned_model)



if __name__ == "__main__":

    main()

