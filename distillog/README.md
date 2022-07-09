# IoTLogAbnormal

- In prunning, we load the teacher model which is trained previously and apply prunning methods. Run prune.py
- In kd:
    + train.py: train the teacher model and noKD model (model with same architecture with student but train from scratch, without Knowledge Distillation)
    + teach.py: distil and transfer from teacher model to student model
    + test.py


