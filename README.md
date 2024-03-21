# DSR
Code For paper "Non-Exemplar Online Class-incremental Continual Learning via Dual-prototype Self-augment and Refinement"

## Usage

### Requirements
requirements.txt

### Data preparation
- CIFAR100 will be downloaded during the first run. (/datasets/cifar100)
- CORE50 download: `source fetch_data_setup.sh`
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download, and place it in datasets/mini_imagenet/


## Run commands
Detailed descriptions of options can be found in [general_main.py](general_main.py)

### CIFAR-100
base_train
```shell
  python general_main.py --agent DSR --base_class 60 --base_epoch 100 --num_task 11 --learning_rate 0.1 --fix_order False --base_batch 100 --num_runs 1 --cl_type nc --data cifar100
 ```
no-example online continual learning
```shell
  python general_main.py --agent DSR --base_class 60 --resume True --num_task 11 --learning_rate 0.01 --fix_order False --num_runs 10 --cl_type nc --data cifar100
 ```

 
 ## Reference
[online-continual-learning](https://github.com/RaptorMai/online-continual-learning)
