#!/bin/bash
cd ../src/
# Example for CIFAR10
python main.py --data_name CIFAR10 --model_name wresnet28x2 --num_experiments 1 --control_name 4000_0.95_100_0.1_non-iid-d-0.3_non-iid-d-0.3_5-5_0.5
python main.py --data_name CIFAR10 --model_name wresnet28x2 --num_experiments 1 --control_name 4000_0.95_100_0.1_non-iid-d-0.1_non-iid-d-0.1_5-5_0.5
python main.py --data_name CIFAR10 --model_name wresnet28x2 --num_experiments 1 --control_name 4000_0.95_100_0.1_iid_iid_5-5_0.5


# Example for SVHN
python main.py --data_name SVHN --model_name wresnet28x2 --num_experiments 1 --control_name 4000_0.95_100_0.1_non-iid-d-0.3_non-iid-d-0.3_5-5_0.5
python main.py --data_name SVHN --model_name wresnet28x2 --num_experiments 1 --control_name 4000_0.95_100_0.1_non-iid-d-0.1_non-iid-d-0.1_5-5_0.5
python main.py --data_name SVHN --model_name wresnet28x2 --num_experiments 1 --control_name 1000_0.95_100_0.1_iid_iid_5-5_0.5

# Example for CIFAR100
python main.py --data_name CIFAR100 --model_name wresnet28x2 --num_experiments 1 --control_name 10000_0.95_100_0.1_non-iid-d-0.3_non-iid-d-0.3_5-5_0.5
python main.py --data_name CIFAR100 --model_name wresnet28x2 --num_experiments 1 --control_name 10000_0.95_100_0.1_non-iid-d-0.1_non-iid-d-0.1_5-5_0.5
python main.py --data_name CIFAR100 --model_name wresnet28x2 --num_experiments 1 --control_name 10000_0.95_100_0.1_iid_iid_5-5_0.5