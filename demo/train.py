import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

num_classes = 1 
num_epochs = 10
batch_size = 16
learning_rate = 0.001
input_size = 20
sequence_length = 300 #300 hoac 100
hidden_size = 128
num_layers = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('../datasets/HDFS/hdfs_vector.json') as f:
    gdp_list = json.load(f)
    value = list(gdp_list.values())


    # Step1-2 PCA: Dimensionality reduction to 20-dimensional data
    from sklearn.decomposition import PCA
    estimator = PCA(n_components=20)
    pca_result = estimator.fit_transform(value)

    # Step1-3 PPA: De-averaged
    ppa_result = []
    result = pca_result - np.mean(pca_result)
    pca = PCA(n_components=20)
    pca_result = pca.fit_transform(result)
    U = pca.components_
    for i, x in enumerate(result):
        for u in U[0:7]:
            x = x - np.dot(u.transpose(), x) * u
        ppa_result.append(list(x))
    ppa_result = np.array(ppa_result)
    #print(ppa_result.shape) (29,20)
 
'''
    Step2: Read training data
'''
def read_data(path,split = 0.7): #chinh split tuy anh chon data lon hay be
    logs_series = pd.read_csv(path)
    logs_series = logs_series.values
    label = logs_series[:,1]
    logs_data = logs_series[:,0]
    logs = []
    for i in range(0,len(logs_data)):
        padding = np.zeros((sequence_length,20))  
        data = logs_data[i]
        data = [int(n) for n in data.split()]
        if sequence_length == 100:
            if len(data) > 100:
                data = data[-100:]
        for j in range(0,len(data)):
            padding[j] = ppa_result[data[j]]
        padding = list(padding)
        logs.append(padding)
    logs = np.array(logs)

    split_boundary = int(logs.shape[0] * split)
    train_x = logs[: split_boundary]
    valid_x = logs[split_boundary:]
    train_y = label[: split_boundary]
    valid_y = label[split_boundary:]
    train_x = np.reshape(train_x, (train_x.shape[0], -1, 20))
    valid_x = np.reshape(valid_x, (valid_x.shape[0], -1, 20))
    train_y = train_y.astype(int)
    valid_y = valid_y.astype(int)

    return train_x, train_y, valid_x, valid_y

train_path = '../datasets/HDFS/log_train.csv'
train_x,train_y,valid_x,valid_y = read_data(train_path)

tensor_x = torch.Tensor(train_x) # transform to torch tensor
tensor_y = torch.from_numpy(train_y)
train_dataset = TensorDataset(tensor_x,tensor_y) # create your dataset
train_loader = DataLoader(train_dataset) # create your dataloader

tensor_valx = torch.Tensor(valid_x) # transform to torch tensor
tensor_valy = torch.from_numpy(valid_y)
test_dataset = TensorDataset(tensor_valx,tensor_valy) # create your dataset
test_loader = DataLoader(test_dataset) # create your dataloader



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
        return out.view(out.size()[0])

model = GRU(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_x)
model.train()
for epoch in range(num_epochs):
     for i, (train_x, train_y) in enumerate(train_loader):  

        # input shape [N, 300, 20]

        train_x = train_x.to(device)
        train_y = train_y.to(device)
         
        # Backward and optimize
        optimizer.zero_grad()
        outputs = model(train_x)
        train_y = train_y.type_as(outputs)
        loss = criterion(outputs, train_y) #target = train_y
       # print(outputs,train_y,loss)
        loss.backward()
        optimizer.step()

        if (i+1) % 500 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    model.eval()
    for log, label in test_loader:
        #log = log.reshape(-1, sequence_length, input_size).to(device)
        log = log.to(device)
        label = label.to(device)
        #print(log, label)
        outputs = model(log)
        label = label.type_as(outputs)
       # print(outputs, label)
        n_samples += label.size(0)
        n_correct += ((outputs > 0.5) == label).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc:.4f} %')
