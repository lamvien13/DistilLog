import json
from multiprocessing.spawn import prepare
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
from utils import LSTM, prepare_data, load_data, train, test, save_model, load_model, apply_weight_sharing 


num_classes = 2
num_epochs = 500
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


def iterative_pruning_finetuning(model, learning_rate, num_epochs, save_path,
                                 lstm_prune_amount,
                                 linear_prune_amount,
                                 num_iter):
    for i in range(0, num_iter):
        print("Pruning and Finetuning {}/{}".format(i + 1, num_iter))
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
        train_x, train_y = prepare_data(train_path, sequence_length)
        train_loader = load_data(train_x, train_y, batch_size)
        model = train(model, train_loader, learning_rate, num_epochs)

        save_model(model, save_path)
        model = load_model(model, save_path)

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
#prune.remove(module, 'weight')
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LSTM):
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

    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    model = load_model(model, save_path)

    test_x, test_y = prepare_data(test_path, sequence_length)
    test_loader = load_data(test_x, test_y, batch_size = batch_size)

    start_time = time()
    accuracy, test_loss, P, R, F1, TP, FP, TN, FN = test(model, test_loader, criterion = nn.CrossEntropyLoss())
    test_loss /= (len(test_loader))

    print('### Before prunning ###')
    print('false positive (FP): {}, false negative (FN): {}, true positive (TP): {}, true negative (TN): {}'.format(FP, FN, TP, TN))
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%). Total time = {time() - start_time}')
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(P, R, F1))

    print("Iterative Pruning + Fine-Tuning...")
    pruned_model = copy.deepcopy(model)

    pruned_model = iterative_pruning_finetuning(pruned_model, learning_rate, num_epochs, save_path,
                                 lstm_prune_amount = 0.04,
                                 linear_prune_amount = 0.03,
                                 num_iter = 15)
    remove_parameters(pruned_model)
    save_model(pruned_model, save_pruned_path)
    print("Finished Prunning")

    ###TEST THE MODEL AFTER PRUNNING


    print('### AFTER PRUNNING ###')
    start_time = time()
    accuracy, test_loss, P, R, F1, TP, FP, TN, FN = test(pruned_model, test_loader, criterion = nn.CrossEntropyLoss())
    test_loss /= (len(test_loader))

    print('false positive (FP): {}, false negative (FN): {}, true positive (TP): {}, true negative (TN): {}'.format(FP, FN, TP, TN))
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%). Total time = {time() - start_time}')
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(P, R, F1))

    apply_weight_sharing(pruned_model)
    
    print('### AFTER WEIGHT SHARING###')
    start_time = time()
    accuracy, test_loss, P, R, F1, TP, FP, TN, FN = test(pruned_model, test_loader, criterion = nn.CrossEntropyLoss())
    test_loss /= (len(test_loader))

    print('false positive (FP): {}, false negative (FN): {}, true positive (TP): {}, true negative (TN): {}'.format(FP, FN, TP, TN))
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%). Total time = {time() - start_time}')
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(P, R, F1))


    save_model(pruned_model, save_pruned_path)  
    print_nonzeros(pruned_model)


if __name__ == "__main__":

    main()

