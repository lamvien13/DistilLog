import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

num_classes = 2
num_epochs = 60
batch_size = 100
learning_rate = 0.001
input_size = 300
sequence_length = 100
hidden_size = 128
num_layers = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('../datasets/HDFS/hdfs_vector.json') as f:
    gdp_list = json.load(f)
    value = list(gdp_list.values())
    vec = []
    for i in range(0, len(value)):
        vec.append(value[i])


def read_data(path):
    logs_series = pd.read_csv(path)
    logs_series = logs_series.values
    label = logs_series[:,1]
    logs_data = logs_series[:,0]
    logs = []
    for i in range(0,len(logs_data)):
        padding = np.full((100,300),-1)         
        data = logs_data[i]
        data = [int(n) for n in data.split()]
        if len(data) > 100:
          data = data[-100:]
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

train_path = '../datasets/HDFS/log_train.csv'
train_x,train_y = read_data(train_path)

tensor_x = torch.Tensor(train_x) # transform to torch tensor
tensor_y = torch.from_numpy(train_y)
train_dataset = TensorDataset(tensor_x,tensor_y) # create your dataset
train_loader = DataLoader(train_dataset, batch_size = batch_size) # create your dataloader

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

model = GRU(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_loader)
model.train()
for epoch in range(num_epochs):
     for i, (train_x, train_y) in enumerate(train_loader):  

        # input shape [N, 300, 20]

        train_x = train_x.to(device)
        train_y = train_y.to(device)
         
        # Backward and optimize
        optimizer.zero_grad()
        outputs = model(train_x)
       # train_y = train_y.type_as(outputs)
        loss = criterion(outputs, train_y) #target = train_y
       # print(outputs,train_y,loss)
        loss.backward()
        optimizer.step()

        print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

######
#TEST#

test_path = '../datasets/HDFS/log_test.csv'
test_x, test_y = read_data(test_path)

tensor_x = torch.Tensor(test_x) # transform to torch tensor
tensor_y = torch.from_numpy(test_y)
test_dataset = TensorDataset(tensor_x,tensor_y) # create your dataset
test_loader = DataLoader(test_dataset, batch_size = batch_size) # create your dataloader


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    TP = 0 
    FP = 0
    FN = 0 
    TN = 0
    for log, label in test_loader:
        log = log.reshape(-1, sequence_length, input_size).to(device)
        label = label.to(device)
        outputs = model(log)
        #print(outputs)
        # max returns (value ,index)
        outputs = F.sigmoid(outputs)[:, 0].cpu().detach().numpy()
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
