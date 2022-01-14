'''
LSTM_model with bidirectional
'''
import torch
import torch.nn as nn
import torchvision

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = 0.5, bidirectional = True, batch_first=True)
        #fully connected layer
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) 

        out, _ = self.lstm(x, (h0,c0))      
        out = out[:, -1, :]
        out = self.fc(out)
        return out