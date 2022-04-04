#xu ly 2 file data va label lai chi con 1 file thoi 

import pandas as pd
import numpy as np

pre_data = pd.read_csv('bgl_data.csv')
pre_label = pd.read_csv('bgl_label.csv')

pre_data = pre_data.values
pre_label = pre_label.values

data = []
label = []

for i in range(0,len(pre_data)):
  for j in range(300):
    value = ''
    value+= str(int(pre_data[i][j])) + ' '

  data.append(value)
  label.append(int(pre_label[i]))

BGL = pd.DataFrame(columns=['sequence','label'])
BGL['sequence'] = data
BGL['label'] = label
BGL.to_csv('log.csv', index=False, header=False)