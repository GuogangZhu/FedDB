# Estimating before Debiasing: A Bayesian Approach to Detaching Prior Bias in Federated Semi-Supervised Learning
This is the PyTorch implemention of our paper Estimating before Debiasing: A Bayesian Approach to Detaching Prior Bias in Federated Semi-Supervised Learning, in IJCAI 2024.

Paper link：https://arxiv.org/abs/2405.19789
## Abstract

Federated Semi-Supervised Learning (FSSL) leverages both labeled and unlabeled data on clients to collaboratively train a model. 
In FSSL, the heterogeneous data can introduce prediction bias into the model, causing the model's prediction to skew towards some certain classes.  Existing FSSL methods primarily tackle this issue by enhancing consistency in model parameters or outputs. However, as the models themselves are biased, merely constraining their consistency is not sufficient to alleviate prediction bias. In this paper, we explore this bias from a Bayesian perspective and demonstrate that it principally originates from label prior bias within the training data. Building upon this insight, we propose a debiasing method for FSSL named FedDB. FedDB utilizes the Average Prediction Probability of Unlabeled Data (APP-U) to approximate the biased prior. During local training, FedDB employs APP-U to refine pseudo-labeling through Bayes' theorem, thereby significantly reducing the label prior bias.  Concurrently, during the model aggregation, FedDB uses APP-U from participating clients to formulate unbiased aggregate weights, thereby effectively diminishing bias in the global model.  Experimental results show that FedDB can surpass existing FSSL methods.

![image](https://github.com/GuogangZhu/FedDB/blob/master/fig/FedDB.jpg)

## Download

```
git clone https://github.com/GuogangZhu/FedDB.git FedDB
```

## Setup

See the `requirements.txt` for environment configuration.

```python
pip install -r requirements.txt
```

## Examples

### CIFAR-10

- **IID**

  ```
  cd ./src
  python main.py --data_name CIFAR10 --model_name wresnet28x2 --num_experiments 1 --control_name 4000_0.95_100_0.1_iid_iid_5-5_0.5
  ```

- **Non-IID with Dir(0.3)**

  ```
  cd ./src
  python main.py --data_name CIFAR10 --model_name wresnet28x2 --num_experiments 1 --control_name 4000_0.95_100_0.1_non-iid-d-0.3_non-iid-d-0.3_5-5_0.5
  ```

- **Non-IID with Dir(0.1)**

  ```
  cd ./src
  python main.py --data_name CIFAR10 --model_name wresnet28x2 --num_experiments 1 --control_name 4000_0.95_100_0.1_non-iid-d-0.1_non-iid-d-0.1_5-5_0.5
  ```

  

### SVHN

- **IID**

  ```
  cd ./src
  python main.py --data_name SVHN --model_name wresnet28x2 --num_experiments 1 --control_name 1000_0.95_100_0.1_iid_iid_5-5_0.5
  ```

  

- **Non-IID with Dir(0.3)**

  ```
  cd ./src
  python main.py --data_name SVHN --model_name wresnet28x2 --num_experiments 1 --control_name 4000_0.95_100_0.1_non-iid-d-0.3_non-iid-d-0.3_5-5_0.5
  ```

  

- **Non-IID with Dir(0.1)**

  ```
  cd ./src
  python main.py --data_name SVHN --model_name wresnet28x2 --num_experiments 1 --control_name 4000_0.95_100_0.1_non-iid-d-0.1_non-iid-d-0.1_5-5_0.5
  ```

  

### CIFAR-100

- **IID**

  ```
  cd ./src
  python main.py --data_name CIFAR100 --model_name wresnet28x2 --num_experiments 1 --control_name 10000_0.95_100_0.1_iid_iid_5-5_0.5
  ```

  

- **Non-IID with Dir(0.3)**

  ```
  cd ./src
  python main.py --data_name CIFAR100 --model_name wresnet28x2 --num_experiments 1 --control_name 10000_0.95_100_0.1_non-iid-d-0.3_non-iid-d-0.3_5-5_0.5
  ```

  

- **Non-IID with Dir(0.1)**

  ```
  cd ./src
  python main.py --data_name CIFAR100 --model_name wresnet28x2 --num_experiments 1 --control_name 10000_0.95_100_0.1_non-iid-d-0.1_non-iid-d-0.1_5-5_0.5
  ```


### Description of control_name

The control_name is pre-defined as follows:

{labeled_num}\_{threshold}\_{client_num}\_{activate_ratio}\_{labeled_distribution}\_{unlabeled_distribution}\_{global_momentum}

- **labeled_num:** number of labled samples

- **threshold:** threshold for pseudo-labeling

- **client_num:** total number of clients

- **activate_ratio:** ratio of clients activated per round

- **labeled_distribution:** use 'iid' to generate independent identically distributed across clients, use 'non-iid-d-0.3' to generate dirichlet non-iid data with $\delta=0.3$ 

- **unlabeled_distribution:** use 'iid' to generate independent identically distributed across clients, use 'non-iid-d-0.3' to generate dirichlet non-iid data with $\delta=0.3$

- **global_momentum:** momentum for server training

These arguments can also be modified in the `config.yml` file.
## Citation

If you make advantage of FedDB in your research, please cite the following in your manuscript:

```latex
@inproceedings{FedDB,
  title={Estimating before Debiasing: A Bayesian Approach to Detaching Prior Bias in Federated Semi-Supervised Learning},
  author={Guogang Zhu, Xufeng Liu, Xinghao Wu, Shaojie Tang, Chao Tang, Jianwei Niu, Hao Su},
  booktitle={Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},
  year={2024}
}
```

## Acknowledgements

This code is modified from https://github.com/diaoenmao/SemiFL-Semi-Supervised-Federated-Learning-for-Unlabeled-Clients-with-Alternate-Training.
