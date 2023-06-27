import argparse
import random
import numpy as np
import torch
import os
from scores import get_score_func
from scipy import stats

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import os

# import models.dataset as dataset
# import models.VGG as VGG
import software_vgg as soft_vgg
# import vgg_imagenet100 as vgg_i100
# from pycls.models.nas.nas import Cell
from utils import add_dropout, init_network
from datetime import datetime
# from torch.cuda.amp import autocast


def get_batch_jacobian(net, x, target, device):
    net.zero_grad()
    x.requires_grad_(True)
    # with autocast():
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return y

def get_dataset(dataset, batch_size):
    if dataset == 'cifar10':
        data_root = '../cifar10_train/pytorch-cifar/data'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)


    return train_loader

def compute_network_score(network, dataset='cifar10', maxofn=1, dropout=True, sigma=0.05, init='',score='hook_logdet', batch_size=128, device='cuda', seed=1):
    # network = soft_vgg.VGG_soft('Custom_2')

    # if dataset == 'cifar10':
    train_loader = get_dataset(dataset, batch_size)
    # else:
    #     repeat = 5
    #     search_batchsize = 128
    #     torch.manual_seed(0)
    #     traindir = os.path.join('/gpfs/gibbs/project/panda/shared/imagenet-100/train')
    #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])
    #
    #     trainset = datasets.ImageFolder(
    #         traindir,
    #         transforms.Compose([
    #             transforms.RandomResizedCrop(224),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             normalize]))
    #
    #     train_loader = torch.utils.data.DataLoader(
    #         trainset, batch_size=search_batchsize, shuffle=False,
    #         num_workers=4, pin_memory=True)


        # testset = torchvision.datasets.CIFAR10(
        #     root=data_root, train=False, download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=100, shuffle=False, num_workers=2)

    try:
        if dropout:
            add_dropout(network, sigma)
        if init != '':
            init_network(network, init)
        if 'hook_' in score:
            network.K = np.zeros((batch_size, batch_size))
            def counting_forward_hook(module, inp, out):
                try:
                    # print(module)
                    # if not module.visited_backwards:
                    #     return
                    if isinstance(inp, tuple):
                        inp = inp[0]
                        # print(inp.size())
                    # print('fwd hook')
                    inp = inp.view(inp.size(0), -1)

                    x = (inp > 0).float()
                    # print(x)
                    K = x @ x.t()
                    # print(f' K {K}')
                    K2 = (1. -x) @ (1. -x.t())
                    # print(f' K2 {K2}')
                    # print(network.K)
                    network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()

                    # print(network.K)
                except Exception as e:
                    print(e)
                    print('problem')
                    pass


            def counting_backward_hook(module, inp, out):
                module.visited_backwards = True


            for name, module in network.named_modules():
                if 'ReLU' in str(type(module)):
                    # hooks[name] = module.register_forward_hook(counting_hook)
                    module.register_forward_hook(counting_forward_hook)
                    module.register_backward_hook(counting_backward_hook)

        network = network.to(device)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        s = []
        for j in range(maxofn):
            data_iterator = iter(train_loader)
            x, target = next(data_iterator)
            x2 = torch.clone(x)
            x2 = x2.to(device)
            x, target = x.to(device), target.to(device)
            # jacobs, labels, y, out = get_batch_jacobian(network, x, target, device, args)
            y = get_batch_jacobian(network, x, target, device)

            if 'hook_' in score:
                network(x2.to(device))
                s.append(get_score_func(score)(network.K, target))
            else:
                s.append(get_score_func(score)(jacobs, labels))
        network_score = np.mean(s)

    except Exception as e:
        print(network)
        print(e)
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, trainval)
        network_score = np.nan

    return network_score



# layers = [torch.nn.Conv2d(in_ch, int(o_ch.item()))]
# network = soft_vgg.VGG_soft('SNN_3')

# network = vgg_i100.VGG_soft('VGG16')
# print(compute_network_score(network, dataset= 'cifar10'))