import argparse
import models
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg, process_args
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset, separate_dataset_su, \
    make_batchnorm_dataset_su
from metrics import Metric
from modules import Server, Client
from utils import to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
import wandb
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

def make_server(model):
    server = Server(model)
    return server


def make_client(model, labeled_data_split, unlabeled_data_split):
    client_id = torch.arange(cfg['num_clients'])
    client = [None for _ in range(cfg['num_clients'])]
    for m in range(len(client)):
        client[m] = Client(client_id[m], model, {'train': labeled_data_split['train'][m], 'test': labeled_data_split['test'][m]}, {'train': unlabeled_data_split['train'][m], 'test': unlabeled_data_split['test'][m]})
    return client


def train_client(labeled_dataset, unlabeled_dataset, server, client, optimizer, metric, logger):
    num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
    for i in range(num_active_clients):
        client[client_id[i]].active = True
    server.distribute(client)
    num_active_clients = len(client_id)
    lr = optimizer.param_groups[0]['lr']
    for i in range(num_active_clients):
        m = client_id[i]
        # adding labeled data to the unlabeled dataset by discarding the label
        dataset_l = separate_dataset(labeled_dataset, client[m].labeled_data_split['train'])
        dataset_u = separate_dataset(unlabeled_dataset, client[m].labeled_data_split['train'] + client[m].unlabeled_data_split['train'])
        # pseudo labeling for unlabeled dataset
        dataset_u = client[m].make_dataset(dataset_u, metric, logger)
        # local training
        client[m].active = True
        client[m].train(dataset_l, dataset_u, lr, metric, logger)
    return

def test(data_loader, model, metric, logger):
    test_loss = 0
    test_acc = 0
    acc = np.array([0 for i in range(cfg['target_size'])])
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            test_loss += evaluation['Loss']*input_size
            test_acc += evaluation['Accuracy']*input_size
            acc += np.array(evaluation['CAccuracy'])
    logger['test_loss'] = test_loss / len(data_loader.dataset)
    logger['test_acc'] = (test_acc / len(data_loader.dataset))
    return acc / np.bincount(data_loader.dataset.target) * 100

def reduce_log(logger):
    logger['train_acc'] = np.mean(logger['train_acc'])
    logger['sl_loss'] = np.mean(logger['sl_loss'])
    logger['ul_loss'] = np.mean(logger['ul_loss'])
    logger['pseudo_acc'] = np.mean(logger['pseudo_acc'])
    logger['mask_pseudo_acc'] = np.mean(logger['mask_pseudo_acc'])
    logger['pseudo_ratio'] = np.mean(logger['pseudo_ratio'])
    return logger

def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    # wandb login
    os.environ['WANDB_MODE'] = 'disabled'
    # replace wandbkey to your own key for wandb login
    wandbkey = ""
    wandb.login(key=wandbkey)
    wandb.init(project="FedDB", entity=cfg['model_tag'], config=cfg)

    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    labeled_dataset = fetch_dataset(cfg['data_name'])
    unlabeled_dataset = fetch_dataset(cfg['data_name'])
    process_dataset(labeled_dataset)
    # generate labeled and unlabeled dataset
    labeled_dataset['train'], unlabeled_dataset['train'], supervised_idx = separate_dataset_su(labeled_dataset['train'],
                                                                                           unlabeled_dataset['train'])
    data_loader = make_data_loader(labeled_dataset, 'global')
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model.parameters(), 'local')
    scheduler = make_scheduler(optimizer, 'global')

    labeled_data_split = split_dataset(labeled_dataset, cfg['num_clients'], cfg['labeled_data_split_mode'])
    unlabeled_data_split = split_dataset(unlabeled_dataset, cfg['num_clients'], cfg['unlabeled_data_split_mode'])

    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy','CAccuracy']})

    last_epoch = 0
    server = make_server(model)
    client = make_client(model, labeled_data_split, unlabeled_data_split)

    # start training
    for epoch in range(last_epoch, cfg['global']['num_epochs'] ):
        logger = {
            'train_acc': [],
            'sl_loss': [],
            'ul_loss': [],
            'pseudo_acc': [], # accuracy of pseudo labels
            'mask_pseudo_acc': [], # accuracy of masked pseudo labels
            'pseudo_ratio': [],
            'test_acc': 0,
            'test_loss': 0,
        }
        # local training on clients
        train_client(labeled_dataset['train'], unlabeled_dataset['train'], server, client, optimizer, metric, logger)
        # debiased model aggregation
        result = server.update(client)
        scheduler.step()
        model.load_state_dict(server.model_state_dict)
        class_acc = test(data_loader['test'], model, metric, logger)

        if (epoch+1) % 50 == 0:
            if not os.path.exists('./output/model'):
                os.makedirs('./output/model')
            torch.save(result, './output/model/{}_epoch_{}.pth'.format(cfg['model_tag'], epoch))

        logger = reduce_log(logger)
        wandb.log(logger)
        print('epoch: {}, train acc {:.2f}%, sl loss {:.2f}, ul loss {:.2f}, pseudo acc {:.2f}%, mask pseudo acc {:.2f}%, pseudo ratio {:.2f}%, test acc {:.2f}%, test loss {:.2f}'.format(
        epoch, logger['train_acc'], logger['sl_loss'], logger['ul_loss'], logger['pseudo_acc'], logger['mask_pseudo_acc'], logger['pseudo_ratio'], logger['test_acc'], logger['test_loss']))
    return

if __name__ == "__main__":
    main()
