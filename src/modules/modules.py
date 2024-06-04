import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from data import make_data_loader, make_batchnorm_stats, FixTransform, MixDataset
from utils import to_device, make_optimizer, collate, to_device
from metrics import Accuracy


class Server:
    def __init__(self, model):
        self.model_state_dict = save_model_state_dict(model.state_dict())
        # optimizer for local training on clients
        optimizer = make_optimizer(model.parameters())
        # optimizer for global aggregation on server
        global_optimizer = make_optimizer(model.parameters(), 'global')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())

    def distribute(self, client, batchnorm_dataset=None):
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict)
        if batchnorm_dataset is not None:
            model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        model_state_dict = save_model_state_dict(model.state_dict())
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
        return

    def update(self, client):
        # updating model on server
        result = {
            'w_global': 0,
            'w_locals': [],
            'client_idx': [],
            'client_ps': []
        }
        # debiased model aggregation
        valid_client = [client[i] for i in range(len(client)) if client[i].active]
        for m in range(len(valid_client)):
            result['w_locals'].append(valid_client[m].model_state_dict)
            result['client_idx'].append(valid_client[m].client_id)
            result['client_ps'].append(valid_client[m].p_s)
        # aggregation weights
        weight = (torch.ones(len(valid_client))/len(valid_client)).reshape((-1,1))
        weight.requires_grad = True
        # APP-U at clients
        client_ps = torch.stack(result['client_ps'])
        # target distribution, set to be a uniform distribution
        target_ps = torch.ones(client_ps.size(1))/client_ps.size(1)
        # updating aggregation weights by SGD
        for i in range(100):
            optimizer = torch.optim.SGD([weight], lr=1.0, momentum=0.9,
                                        weight_decay=0.9, nesterov=True)
            output = (client_ps*weight).sum(0)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(output, target_ps)
            loss.backward()
            optimizer.step()
            weight = torch.softmax(weight.detach(), 0)
            weight.requires_grad = True
        weight = weight.detach().reshape(-1)
        # model aggregation
        with torch.no_grad():
            if len(valid_client) > 0:
                model = eval('models.{}()'.format(cfg['model_name']))
                model.load_state_dict(self.model_state_dict)
                global_optimizer = make_optimizer(model.parameters(), 'global')
                global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                global_optimizer.zero_grad()
                for k, v in model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(valid_client)):
                            tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                global_optimizer.step()
                self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                self.model_state_dict = save_model_state_dict(model.state_dict())
                result['w_global'] = self.model_state_dict
        for i in range(len(client)):
            client[i].active = False
        return result


class Client:
    def __init__(self, client_id, model, labeled_data_split, unlabeled_data_split):
        self.client_id = client_id
        self.labeled_data_split = labeled_data_split
        self.unlabeled_data_split = unlabeled_data_split
        self.model_state_dict = save_model_state_dict(model.state_dict())

        optimizer = make_optimizer(model.parameters(), 'local')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())

        self.active = False
        self.beta = torch.distributions.beta.Beta(torch.tensor([cfg['alpha']]), torch.tensor([cfg['alpha']]))
        self.verbose = cfg['verbose']
        self.lps = []
        self.b_p_s = []
        self.a_p_s = []

    def make_hard_pseudo_label(self, soft_pseudo_label, tau = None):
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        if tau!=None:
            mask = []
            for i in range(len(hard_pseudo_label)):
                mask.append(max_p[i]>tau[hard_pseudo_label[i]])
            mask = torch.tensor(mask)
        else:
            mask = max_p.ge(cfg['threshold'])
        return hard_pseudo_label, mask

    def get_p_s(self, dataset):
        data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict, strict=False)
        p_s = torch.zeros(10).cuda()
        with torch.no_grad():
            model.train(False)
            for i, input in enumerate(data_loader):
                input = collate(input)
                input = to_device(input, cfg['device'])
                output = model(input)
                prob = torch.softmax(output['target'], dim=1)
                p_s += prob.sum(dim=0)
        return (p_s/len(data_loader.dataset)).cpu().numpy()
    
    def make_dataset(self, dataset, metric, logger):
        with torch.no_grad():
            data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            model.train(False)
            # model output
            output = []
            # ground truth label
            target = []
            for i, input in enumerate(data_loader):
                input = collate(input)
                input = to_device(input, cfg['device'])
                output_ = model(input)
                output_i = output_['target']
                target_i = input['target']
                output.append(output_i.cpu())
                target.append(target_i.cpu())
            output_, input_ = {}, {}
            output_['target'] = torch.cat(output, dim=0)
            output_['target'] = F.softmax(output_['target'], dim=-1)
            # averaged prediction probabilities
            self.p_s = output_['target'].mean(0)
            # debiased pseudo labeling
            output_['target'] = (output_['target']/self.p_s)/((output_['target']/self.p_s).sum(1)).reshape(-1,1)
            input_['target'] = torch.cat(target, dim=0)

            new_target, mask = self.make_hard_pseudo_label(output_['target'])
            output_['mask'] = mask
            evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
            logger['pseudo_acc'].append(evaluation['PAccuracy'])
            logger['mask_pseudo_acc'].append(evaluation['MAccuracy'])
            logger['pseudo_ratio'].append(evaluation['LabelRatio'])
            if torch.any(mask):
                fix_dataset = copy.deepcopy(dataset)
                fix_dataset.target = new_target.tolist()
                mask = mask.tolist()
                fix_dataset.data = list(compress(fix_dataset.data, mask))
                fix_dataset.target = list(compress(fix_dataset.target, mask))
                fix_dataset.other = {'id': list(range(len(fix_dataset.data)))}

                mix_dataset = None
                return fix_dataset, mix_dataset

            else:
                return None

    def train(self, dataset_l, dataset_u, lr, metric, logger):
        if dataset_u==None:
            data_loader = make_data_loader({'train': dataset_l}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = 'sup'
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation_l = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger['sl_loss'].append(evaluation_l['Loss'])
                    logger['train_acc'].append(evaluation_l['Accuracy'])
                    if dataset_u != None:
                        evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                        logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        else:
            fix_dataset, _ = dataset_u
            sl_batch_size = len(dataset_l)
            ul_batch_size = len(fix_dataset)
            data_loader = make_data_loader({'train': dataset_l}, 'client', batch_size={'train':sl_batch_size})['train']
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client', batch_size={'train':ul_batch_size})['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for (i, input_l),(i_u, input_u) in zip(enumerate(data_loader), enumerate(fix_data_loader)):
                    input_l = collate(input_l)
                    input_u = collate(input_u)
                    input_l['loss_mode'] = 'sup'
                    input_u['loss_mode'] = 'semi'
                    input_l = to_device(input_l, cfg['device'])
                    input_u = to_device(input_u, cfg['device'])
                    optimizer.zero_grad()
                    output_l = model(input_l)
                    output_u = model(input_u)
                    # momentum update of APP-U for stability
                    self.p_s = 0.5 * self.p_s + 0.5 * F.softmax(output_u['target'], dim=-1).mean(0).cpu().detach()
                    loss = output_l['loss'] + output_u['loss']
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation_l = metric.evaluate(['Loss', 'Accuracy'], input_l, output_l)
                    evaluation_u = metric.evaluate(['Loss'], input_u, output_u)
                    logger['sl_loss'].append(evaluation_l['Loss'])
                    logger['train_acc'].append(evaluation_l['Accuracy'])
                    logger['ul_loss'].append(evaluation_u['Loss'])
                    if num_batches is not None and i == num_batches - 1:
                        break

        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        return


def save_model_state_dict(model_state_dict):
    return {k: v.cpu() for k, v in model_state_dict.items()}


def save_optimizer_state_dict(optimizer_state_dict):
    optimizer_state_dict_ = {}
    for k, v in optimizer_state_dict.items():
        if k == 'state':
            optimizer_state_dict_[k] = to_device(optimizer_state_dict[k], 'cpu')
        else:
            optimizer_state_dict_[k] = copy.deepcopy(optimizer_state_dict[k])
    return optimizer_state_dict_
