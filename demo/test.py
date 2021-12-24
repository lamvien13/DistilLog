import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


num_classes = 2
num_epochs = 60
batch_size = 100
learning_rate = 0.001
input_size = 300
sequence_length = 50
hidden_size = 128
num_layers = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('../datasets/HDFS/hdfs_vector.json') as f:
    gdp_list = json.load(f)
    value = list(gdp_list.values())
    vec = []
    for i in range(0, len(value)):
        vec.append(value[i])


def read_data(path, i):
    logs_series = pd.read_csv(path)
    tqdm.pandas(desc="Loading")
    logs_series.progress_apply(lambda x: x)
    logs_series = logs_series.values

    total = len(logs_series)
    sub = int(total/50)
    if i==49:
        label = logs_series[49*sub:,1]
        logs_data = logs_series[49*sub:,0]
    else:
        label = logs_series[i*sub:(i+1)*sub,1]
        logs_data = logs_series[i*sub:(i+1)*sub,0]
    
    
    logs = []
    for i in tqdm (range(0,sub), desc="Loading batch"):
    #for i in range(0,len(logs_data)):
        padding = np.full((50,300),-1)         
        data = logs_data[i]
        data = [int(n) for n in data.split()]
        if len(data) > 50:
          data = data[-50:]
        for j in range(0,len(data)):
            padding[j] = vec[data[j]]
        padding = list(padding)
        logs.append(padding)
    logs = np.array(logs)
    train_x = logs
    train_y = label
    train_x = np.reshape(train_x, (train_x.shape[0], -1, 300))
    train_y = train_y.astype(int)

    return train_x, train_y

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
       
        # x: (n, 300, 20), h0: (2, n, 128)
        
        # Forward propagate RNN
        out, _ = self.gru(x, h0)      
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


######
#TEST#

test_path = '../datasets/HDFS/log_test.csv'
save_path = '../datasets/HDFS/model.pth'


def load_data(test_x, test_y):
    tensor_x = torch.Tensor(test_x) # transform to torch tensor
    tensor_y = torch.from_numpy(test_y)
    test_dataset = TensorDataset(tensor_x,tensor_y) # create your dataset
    test_loader = DataLoader(test_dataset, batch_size = batch_size) # create your dataloader

    return test_loader


model = GRU(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load(save_path))
model.to(device)
model.eval()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    TP = 0 
    FP = 0
    FN = 0 
    TN = 0
    for i in range(50):
        test_x, test_y = read_data(test_path, i)
        test_loader = load_data(test_x, test_y)
        tbar = tqdm(test_loader, desc="\r")
        for i, (log, label) in enumerate(tbar):
            log = log.reshape(-1, sequence_length, input_size).to(device)
            label = label.to(device)
            outputs = model(log)
            #print(outputs)
            # max returns (value ,index)
            outputs = torch.sigmoid(outputs)[:, 0].cpu().detach().numpy()
            predicted = (outputs < 0.2).astype(int)
            label = np.array([y.cpu() for y in label])
            #print(predicted, label)
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
