# Dataset

## 1. HDFS

This log set is generated in a private cloud environment using benchmark workloads, and manually labeled through handcrafted rules to identify the anomalies. The logs are sliced into traces according to block ids. Then each trace associated with a specific block id is assigned a groundtruth label: normal/anomaly (available in anomaly_label.csv). You may find more details of this dataset from the original paper:

Wei Xu, Ling Huang, Armando Fox, David Patterson, Michael Jordan. [Detecting Large-Scale System Problems by Mining Console Logs](https://www.sigops.org/s/conferences/sosp/2009/papers/xu-sosp09.pdf), in Proc. of the 22nd ACM Symposium on Operating Systems Principles (SOSP), 2009.

- [Raw data](https://zenodo.org/record/3227177/files/HDFS_1.tar.gz)

- [Preprocessing](https://github.com/vanhoanglepsa/NeuralLog/blob/d5917e319b928a64408f7cafc43c2ac6fecf12e7/neurallog/data_loader.py#L127)

## 2. BGL

BGL is an open dataset of logs collected from a BlueGene/L supercomputer system at Lawrence Livermore National Labs (LLNL) in Livermore, California, with 131,072 processors and 32,768GB memory. The log contains alert and non-alert messages identified by alert category tags. In the first column of the log, "-" indicates non-alert messages while others are alert messages. The label information is amenable to alert detection and prediction research. It has been used in several studies on log parsing, anomaly detection, and failure prediction.

You may find more details of this dataset from the original paper:

Adam J. Oliner, Jon Stearley. [What Supercomputers Say: A Study of Five System Logs](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.97.8608&rep=rep1&type=pdf), in Proc. of IEEE/IFIP International Conference on Dependable Systems and Networks (DSN), 2007.

*Note*: Each log line has its label. "-" label indicates normal log.

- [Raw data](https://zenodo.org/record/3227177/files/BGL.tar.gz)

- [Preprocessing](https://github.com/vanhoanglepsa/NeuralLog/blob/d5917e319b928a64408f7cafc43c2ac6fecf12e7/neurallog/data_loader.py#L300)

