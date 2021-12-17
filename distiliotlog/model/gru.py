'''
GRU_model
'''
import torch
import torch.nn as nn
import torchvision

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, device):
        # Set initial hidden states
        input0 = x[0]
        h0 = torch.zeros(self.num_layers, x.size(0), 
                        self.hidden_size).to(device)        
        # x: (n, 300, _), h0: (2, n, 128)
        
        # Forward propagate RNN
        out, _ = self.gru(x, h0)          
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out, _ = self.gru(input0, h0)
        out = self.fc(out[:, -1, :])
        return out