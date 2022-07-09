import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Parameter
from torch.nn.modules.module import Module
from tqdm import tqdm
import torch.nn.functional as F
import math
from time import time 

from utils import DistilLog, read_data, load_data, load_model, save_model

num_classes = 2
num_epochs = 100
batch_size = 50
input_size = 30
sequence_length = 50
num_layers = 1
hidden_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = ('../datasets/HDFS/train.csv')
save_teacher_path = ('../datasets/HDFS/model/teacher.pth')
save_student_path = ('../datasets/HDFS/model/student.pth')

train_x, train_y = read_data(train_path, input_size, sequence_length)
train_loader = load_data(train_x, train_y, batch_size)


def train_step(
    Teacher,
    Student,
    optimizer,
    student_loss_fn,
    divergence_loss_fn,
    temp,
    alpha,
    epoch,
    device
):
    losses = []
    pbar = tqdm(train_loader, total=len(train_loader), position=0, leave=True, desc=f"Epoch {epoch+1}")
    for data, targets in pbar:
        # Get data to cuda if possible
        data = data.to(device)
        targets = targets.to(device)

        # forward
        with torch.no_grad():
            teacher_preds, _ = Teacher(data)

        student_preds, __ = Student(data)
        student_loss = student_loss_fn(student_preds, targets)
        
        ditillation_loss = divergence_loss_fn(
            F.softmax(student_preds / temp, dim=1),
            F.softmax(teacher_preds / temp, dim=1)
        )
        loss = alpha * student_loss + (1 - alpha) * ditillation_loss
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    
    avg_loss = sum(losses) / len(losses)
    return avg_loss

def teach(epochs, Teacher, Student, temp=7, alpha=0.3):
  Teacher = Teacher.to(device)
  Student = Student.to(device)
  student_loss_fn = nn.CrossEntropyLoss()
  divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
  optimizer = torch.optim.Adam(Student.parameters(), lr=0.01)

  Teacher.eval()
  Student.train()
  for epoch in range(epochs):
      loss = train_step(
          Teacher,
          Student,
          optimizer,
          student_loss_fn,
          divergence_loss_fn,
          temp,
          alpha,
          epoch,
          device
      )

      print(f"Loss:{loss:.2f}")

Teacher = DistilLog(input_size = input_size, hidden_size=128, num_layers = 2, num_classes = num_classes, is_bidirectional=False).to(device)
Student = DistilLog(input_size = input_size, hidden_size=4, num_layers = 1, num_classes = num_classes, is_bidirectional=False).to(device)

Teacher = load_model(Teacher, save_teacher_path)
teach(epochs=100, Teacher=Teacher, Student=Student, temp=7, alpha=0.3)
save_model(Student, save_student_path)
