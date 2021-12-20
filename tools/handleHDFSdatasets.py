import pandas as pd
import copy
from collections import Counter

pre_data = pd.read_csv('./hdfs/HDFS_sequence.csv').values
blk_label = pd.read_csv('./hdfs/anomaly_label.csv').values

data = []
label = []

for i in range(0,len(pre_data)):
    value = ''
    division = pre_data[i][1].split(",")
    if division[0] != '[]':
        for j in range(0,len(division)):
            if '[' in division[j] and ']' not in division[j]:
                value+= str(int(division[j][3:-1])) + ' '
            elif '[' in division[j] and ']' in division[j]:
                value+= str(int(division[j][3:-2])) + ' '
            elif '[' not in division[j] and ']' in division[j]:
                value+= str(int(division[j][3:-2])) + ' '
            else:
                value+= str(int(division[j][3:-1])) + ' '
    else:
        value += '0'
    data.append(value)
    label.append(str(pre_data[i][0])) #label cho nay la blk_id

HDFS_sequence = pd.DataFrame(columns=['sequence','label'])
HDFS_sequence['sequence'] = data
HDFS_sequence['label'] = label
Blockid_label = pd.DataFrame(columns=['label','value'])
Blockid_label['label'] = 
HDFS_sequence.to_csv('hdfs_log_train.csv', index=False, header=False)
