import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.modules.module import Module
from torchinfo import summary
from tqdm import tqdm
import csv
from time import time
from torch.nn import functional as F
from attention_layers import LinearAttention

device = torch.device('cpu')


class DistilLog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, is_bidirectional=True,
                 use_linear_attention=False):
        super(DistilLog, self).__init__()
        self.num_directions = 2 if is_bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=0.1, batch_first=True,
                          bidirectional=is_bidirectional)
        # if use_linear_attention:
        #     self.attn = LinearAttention(tensor_1_dim=)
        # else:
        self.attn = self.attention
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
        self.attention_size = self.hidden_size
        self.w_omega = Variable(
            torch.zeros(self.hidden_size * self.num_directions, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))

    def attention(self, gru_output, seq_len):
        output_reshape = torch.Tensor.reshape(gru_output,
                                              [-1, self.hidden_size * self.num_directions])

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega.to(device)))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega.to(device), [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, seq_len])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, seq_len, 1])
        state = gru_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        gru_output, _ = self.gru(x)
        gru_output = self.attention(gru_output, sequence_length)
        # print(gru_output.shape, hidden.shape)
        # # get last hidden state
        # final_state = hidden.view(self.num_layers, batch_size, self.hidden_size)[-1]
        # final_hidden_state = None
        # final_hidden_state = final_state.squeeze(0)
        #
        # # push through attention layer
        # attn_weights = None
        # # gru_output = gru_output.permute(1, 0, 2)  #
        # x, attn_weights = self.attn(gru_output, final_hidden_state)
        x = self.fc(gru_output)

        return x, x


def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    return model


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def mod(l, n):
    """ Truncate or pad a list """
    r = l[-1 * n:]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def read_data(path, input_size, sequence_length):
    fi = pd.read_csv('../datasets/HDFS/pca_vector.csv', header = None)
    vec = []
    vec = fi
    vec = np.array(vec)

    logs_series = pd.read_csv(path)
    logs_series = logs_series.values
    label = logs_series[:, 1]
    logs_data = logs_series[:, 0]
    logs = []
    for i in range(0, len(logs_data)):
        ori_seq = [
            int(eventid) for eventid in logs_data[i].split()]
        seq_pattern = mod(ori_seq, sequence_length)
        vec_pattern = []

        for event in seq_pattern:
            if event == 0:
                vec_pattern.append([-1] * input_size)
            else:
                vec_pattern.append(vec[event - 1])
        logs.append(vec_pattern)
    logs = np.array(logs)
    train_x = logs
    train_y = np.array(label)
    train_x = np.reshape(train_x, (train_x.shape[0], -1, input_size))
    train_y = train_y.astype(int)

    return train_x, train_y


def load_data(train_x, train_y, batch_size):
    tensor_x = torch.Tensor(train_x)
    tensor_y = torch.from_numpy(train_y)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader


def train(model, train_loader, learning_rate, num_epochs):
    criterion = nn.CrossEntropyLoss()
    summary(model, input_size=(50, 50, 30))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    model.train()

    for epoch in range(num_epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        total_loss = 0
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()


            if (batch_idx + 1) % 10 == 0:
                done = (batch_idx + 1) * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(
                    f'Train Epoch: {epoch + 1}/{num_epochs} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {total_loss:.6f}')

    return model


if __name__ == '__main__':
    model = DistilLog(input_size=30, hidden_size=128, num_layers=2, num_classes=2, is_bidirectional=False)
    inp = torch.rand((1, 8, 30))
    print(inp.shape)
    out, _ = model(inp)
    print(out.shape)
