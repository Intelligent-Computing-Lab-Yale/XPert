import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import copy
import compute_score as score_compute
import os
import time
import numpy as np
from scipy.special import softmax
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=12, type=float, help='learning rate')
parser.add_argument('--hw_params', default='16,4,274,64', help='n_pe_tile, n_xbar_pe, total_tiles, xbar_size')
parser.add_argument('--t_latency', default=5000., type=float, help='target latency')
parser.add_argument('--t_area', default=50e6, type=float, help='target area')
parser.add_argument('--factor', default=0.75, type=float, help='factor for PEs')
parser.add_argument('--epochs', default=2000, type=int, help='Epochs')
parser.add_argument('--wt_prec', default=8, type=int, help='Weight Precision')
parser.add_argument('--cellbit', default=4, type=int, help='Number of Bits/Cell')
parser.add_argument('--area_tolerance', default=5, type=int, help='Area Tolerance (%)')

args = parser.parse_args()
class masking(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a_param):

        m = nn.Softmax()
        p_vals = m(a_param)

        gates = torch.FloatTensor([1]).cuda() * (1. *(p_vals == torch.max(p_vals))).float()

        return gates

    @staticmethod
    def backward(ctx, grad_output):

        grad_return = grad_output
        return grad_return, None

class compute_latency(nn.Module):

    def forward(self, layer_latency_nodes, layer_arch_params):
        softmax = nn.Softmax()
        prob = softmax(layer_arch_params)
        lat = torch.sum(layer_latency_nodes * prob)

        return lat

k_space = [3]
# ch_space = [16, 32, 64, 128, 256, 512]
ch_space = [64, 128, 256, 512]
f_space = [32] #, 16, 8, 4]
p_space = [1, 2, 4, 8, 32, 64]
adc_share_space = [2, 4, 8, 16, 32]
adc_type_space = [1, 2]
## Hw config
n_pe_tile, n_xbar_pe, total_tiles, xbar_size = map(float, args.hw_params.split(',')) #torch.tensor([8]).cuda()

network_latency_nodes = []
network_arch = []
total_layers = 13

for layers in range(total_layers):
    layer_arch = []
    layer_latency_nodes = []
    for adc in adc_type_space:
        for i in ch_space:
            for j in k_space:
                for k in f_space:
                    for l in p_space:
                        for m in adc_share_space:
                            if layers < 2:
                                layer_arch.append((adc, i, j, k, l, m))
                                # latency = (n_pe_tile * k * k * m)/float(l)
                                # layer_latency_nodes.append(latency)
                            elif layers >= 2 and layers < 4:
                                layer_arch.append((adc, i, j, k/2., l, m))
                                # latency = (n_pe_tile * k * k * m) / float(4*l)
                                # layer_latency_nodes.append(latency)
                            elif layers >= 4 and layers < 7:
                                layer_arch.append((adc, i, j, k/4., l, m))
                                # latency = (n_pe_tile * k * k * m) / float(16*l)
                                # layer_latency_nodes.append(latency)
                            elif layers >= 7 and layers < 10:
                                layer_arch.append((adc, i, j, k/8., l, m))

                            elif layers >= 10 and layers < 13:
                                layer_arch.append((adc, i, j, k / 16., l, m))
                                # latency = (n_pe_tile * k * k * m) / float(64*l)
                                # layer_latency_nodes.append(latency)

    # print(len(layer_latency_nodes))
    network_latency_nodes.append(layer_latency_nodes)
    network_arch.append(layer_arch)

def LUT_area(tiles, mux, adc_type):
    if adc_type == 1:
        a,b,c,d,e = 2557.166495434807, 1325750339.0150185, 0.0037802706563706337, -1325744426.1570153, 355177.96577219985
    if adc_type == 2:
        a, b, c, d, e = 1662.1420546233026, 1647434588.0067654, 0.0011610296696243961, -1647433897.2349775, 328615.0508042259

    area = (a * mux + e) * tiles + b * torch.exp(c * tiles / mux) + d

    return area

# def LUT_latency(tiles, mux, adc_type, speedup, feature_size):
#     a, b, c, d, e = -0.0013059562090877996, -176121.8732468568, 0.020443304045075883, 88062.15068343304, 88062.150683433
#     # print(f'tiles {tiles}, mux {mux}')
#
#     latency = (a * mux*mux * tiles + b) + (c * tiles*tiles * mux + d) + e
#     latency = latency * feature_size**2 / speedup
#     if adc_type == 1:
#         latency = latency
#     if adc_type == 2:
#         latency = latency * 16
#
#     return latency

def LUT_latency(tiles, mux, adc_type, speedup, feature_size):
    # a, b, c, d, e = -0.0013059562090877996, -176121.8732468568, 0.020443304045075883, 88062.15068343304, 88062.150683433
    # print(f'tiles {tiles}, mux {mux}')


    if adc_type == 1:
        a, b, c, d, e = 0.08836970306542265, 8.466777271635184e-05, 1.0, 0.10469741426869324, 0.12532326984887984

        # latency = latency
    if adc_type == 2:
        a, b, c, d, e = 0.24051026469311507, 0.000432503086158724, 1.0, 0.4050948974138533, 0.16033299026875378

        # return a * x * e + b * y ** 2 + d
        # latency = latency * 16

    latency = a * tiles * e + b * mux * mux + d
    # latency = (a * mux * mux * tiles + b) + (c * tiles * tiles * mux + d) + e
    latency = latency * feature_size ** 2 / speedup
    # a * x * e + b * y ** 2 + d
    return latency

# def LUT_area(tiles, mux, adc_type):
#     if adc_type == 1:
#         a,b,c,d,e = 2557.166495434807, 1325750339.0150185, 0.0037802706563706337, -1325744426.1570153, 355177.96577219985
#     if adc_type == 2:
#         a, b, c, d, e = 1662.1420546233026, 1647434588.0067654, 0.0011610296696243961, -1647433897.2349775, 328615.0508042259
#
#     area = (a * mux + e) * tiles + b * torch.exp(c * tiles / mux) + d
#
#     return area#*1e-6
#
# def LUT_latency(tiles, mux, adc_type, speedup, feature_size):
#     # a, b, c, d, e = -0.0013059562090877996, -176121.8732468568, 0.020443304045075883, 88062.15068343304, 88062.150683433
#     # print(f'tiles {tiles}, mux {mux}')
#
#
#     if adc_type == 1:
#         a, b, c, d, e = 0.08836970306542265, 8.466777271635184e-05, 1.0, 0.10469741426869324, 0.12532326984887984
#
#         # latency = latency
#     if adc_type == 2:
#         a, b, c, d, e = 0.24051026469311507, 0.000432503086158724, 1.0, 0.4050948974138533, 0.16033299026875378
#
#         # return a * x * e + b * y ** 2 + d
#         # latency = latency * 16
#
#     latency = a * tiles * e + b * mux * mux + d
#     # latency = (a * mux * mux * tiles + b) + (c * tiles * tiles * mux + d) + e
#     latency = latency * feature_size ** 2 / speedup
#     # a * x * e + b * y ** 2 + d
#     return latency#*1e-3

# network_latency_nodes = np.array(network_latency_nodes)
# network_latency_nodes = torch.FloatTensor(network_latency_nodes)
# network_latency_nodes = Variable(network_latency_nodes, requires_grad= True)
# network_latency_nodes = network_latency_nodes.cuda()

network_arch = np.array(network_arch)
network_arch = torch.from_numpy(network_arch)
network_arch = network_arch.cuda()

# print(f' len {network_arch[0])

th0 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th1 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th2 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th3 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th4 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th5 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th6 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th7 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th8 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th9 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th10 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th11 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)
th12 = torch.nn.Parameter(torch.randn(len(layer_arch)).cuda(), requires_grad=True)

arch_params = [th0, th1, th2, th3, th4, th5, th6, th7, th8, th9, th10, th11, th12] #, th12]

optimizer = optim.SGD(arch_params, lr=args.lr, momentum=0.99, weight_decay=0.00001)
# optimizer = optim.Adam(arch_params, lr=args.lr,weight_decay=0.0001)
# scheduler = ExponentialLR(optimizer, gamma=0.9)

loss_fn = torch.nn.MSELoss()
s = nn.Softmax()

target_latency = torch.FloatTensor([args.t_latency])
target_latency = target_latency.cuda()
target_latency = Variable(target_latency, requires_grad = True)

target_util = torch.FloatTensor([1])
target_util = target_util.cuda()
target_util = Variable(target_util, requires_grad = True)
comp_lat = compute_latency()

latency_list = []
arch_param_epoch = []
lat_err_list, r_loss_list, err_list, tile_list = [], [], [], []
best_pe = 0
best_diff = 50e6
best_area = 500e6
best_latency = 5000000.
best_score = 0

pool_layers = [1, 3, 6, 9]

class custom_model(nn.Module):
    def __init__(self, features, classifier):
        super(custom_model, self).__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, x):
        out = self.features(x)
        # print(out.size())
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

best_network = nn.Sequential(*([torch.nn.BatchNorm2d(3)]))
prev_list = [3]
for epochs in range(args.epochs):
    print(epochs)
    total_latency = 0
    total_pe = 0
    if epochs % 100 == 0:
        # print(f'arch param {arch_params[0][0]}')
        a_param = copy.deepcopy(arch_params)
        arch_param_epoch.append(a_param)
    # for i in range(total_layers):

        # lat = comp_lat(network_latency_nodes[i], arch_params[i])
        #
        # total_latency += lat
    # print(f' %%%%%% total latency {total_latency}')
    # error = total_latency
    # error = 0.01*loss_fn(total_latency, target_latency.detach())
    # lat_err_list.append(error.item())
    parallel = []
    mux = []
    adc_arch = []
    if epochs % 1 == 0:
        in_ch = 3
        total_pe = 0
        tile_area = 0
        tile_latency = 0
        t_speedup = 0
        act_tile_area = 0
        act_tile_latency = 0
        expected_util = 0
        abs_tile = 0
        sup, t_use, util_list = [], [], []
        # print(f'arch parame {arch_params}')
        layers = []
        o_list = []
        for i in range(total_layers):
            prob = s(arch_params[i])
            index = torch.argmax(prob)
            a = network_arch[i][index]

            gate = masking.apply(prob)
            # print(f'================ layer  {i} ================')
            adc_type = a[0]
            o_ch = a[1]
            k = a[2]
            f = a[3]
            m = a[5]
            # print(network_arch[i])
            # print(f' gate {gate}')
            # print(f'o_ch {o_ch}, k {k}, p {p}')
            xbar_s = torch.FloatTensor([xbar_size]).cuda()
            total_area = torch.FloatTensor([args.t_area]).cuda()
            # print(f' xbar_size {xbar_s.item(   )}, n_xbar_tile {n_xbar_tile.item()}')
            # pe = torch.ceil(torch.ceil(in_ch/xbar_s)* torch.ceil(o_ch/xbar_s)* k*k*p / n_xbar_tile)

            # network_configs.append((o_ch, k))
            # print(f'i {i}, in_ch {in_ch}, int(o_ch.item()) {int(o_ch.item())}, int(k.item()) {int(k.item())}')
            layers += [torch.nn.Conv2d(in_ch, int(o_ch.item()), kernel_size=int(k.item()), padding=1),
                       torch.nn.BatchNorm2d(int(o_ch.item())),
                       torch.nn.ReLU(inplace=True)]
            o_list.append(int(o_ch.item()))
            if i in pool_layers:
                layers += [torch.nn.MaxPool2d(2,2)]

            if i < total_layers-1:
                # pe = torch.ceil((in_ch*k*k / xbar_s) * o_ch*4 / xbar_s * p / n_xbar_tile)
                # abs_pe = torch.ceil((torch.ceil(in_ch*k*k / xbar_s) * torch.ceil(o_ch*4 / xbar_s) * p / n_xbar_tile))
                # total_pe += torch.ceil(in_ch*k*k / xbar_s) * torch.ceil(o_ch*4 / xbar_s) * p
                # util = pe/abs_pe
                n_cells = int(args.wt_prec / args.cellbit)
                # if (o_ch*n_cells / xbar_s) >= 1:
                #     n_xbars = (torch.ceil(in_ch*k*k / xbar_s) * torch.ceil(o_ch*n_cells / xbar_s) * p)
                # else:
                #     n_xbars = (torch.ceil(in_ch * k * k / xbar_s) * torch.ceil(o_ch * n_cells * p / xbar_s))
                n_xbars = torch.ceil(in_ch*k*k / xbar_s) * torch.ceil(o_ch*n_cells / xbar_s)
                # n_adcs = xbar_s/m * n_xbars
                n_tiles = torch.ceil(n_xbars/(n_pe_tile*n_xbar_pe))
                if (o_ch * n_cells / xbar_s) >= 1:
                    speedup = (n_tiles * n_xbar_pe * n_pe_tile / n_xbars)
                else:
                    speedup = (n_tiles * n_xbar_pe * n_pe_tile / n_xbars) * (xbar_s / (o_ch * n_cells))

                # if (o_ch*n_cells / xbar_s) >= 1:
                #     speedup = (n_tiles * n_xbar_pe * n_pe_tile / n_xbars)
                # else:
                #     speedup = (n_tiles * n_xbar_pe * n_pe_tile / n_xbars) * (xbar_s/(o_ch*n_cells))


                act_tile_area += LUT_area(n_tiles, m, adc_type)
                act_tile_latency += LUT_latency(n_tiles, m, adc_type, speedup, f)

                # act_tile_area += 668468 * (0.9981761863406945 * n_tiles - 0.041827731986675554)
                # 0.015596502909426158 - 0.041827731987578165
                tile_area += gate.sum() * LUT_area(n_tiles, m, adc_type) #(668468 * (0.9981761863406945 * n_tiles - 0.041827731986675554))
                tile_latency += gate.sum() * LUT_latency(n_tiles, m, adc_type, speedup, f)

                # t_speedup += gate.sum() * speedup
                # print(f'gate.sum() {gate.sum()}')
                t_use.append(n_tiles.item())

                # print(f'-------------- layer {i+1} xbar_size {xbar_s.item()}, n_xbar_tile {n_xbar_tile.item()} in_ch {in_ch} k {k} o_ch {o_ch} p {p} pe {pe.item()} abs_pe {abs_pe.item()} total_pe {total_pe.item()}')

            # else:
            #     # pe = torch.ceil(in_ch*k*k/xbar_s* o_ch*4/xbar_s * p / n_xbar_tile)
            #     # abs_pe = torch.ceil(torch.ceil(in_ch * k*k/xbar_s)* torch.ceil(o_ch*4/xbar_s) * p / n_xbar_tile)
            #     # total_pe += torch.ceil(in_ch* k*k/xbar_s)* torch.ceil(o_ch*4/xbar_s)*p
            #     # util = pe/abs_pe
            #     n_xbars = (torch.ceil(in_ch * k * k / xbar_s) * torch.ceil(o_ch * 4 / xbar_s) * p)
            #     n_adcs = xbar_s / m * n_xbars
            #     adc_area += 2.05E+06 * (0.00023615 * n_adcs + 0.033151)
                # print(f'-------------- layer {i+1} xbar_size {xbar_s.item()}, n_xbar_tile {n_xbar_tile.item()} in_ch {in_ch} k {k} o_ch {o_ch} p {p} pe {pe.item()} abs_pe {abs_pe.item()} total_pe {total_pe.item()}')
            # total_pe += pe
            # sup.append(p.item())
            # t_use.append(abs_pe.item())
            # util_list.append(util.item())
            # abs_tile += abs_pe

            # print(f' prob {prob[index]}')
            # expected_pe += gate.sum()*abs_pe
            # expected_util += prob[index]*util

            if i == total_layers-1:
                # print(in_ch)
                feature_size = int(32 / (2**len(pool_layers)))
                n_cells = int(args.wt_prec / args.cellbit)
                fc_xbars = torch.ceil(o_ch / xbar_s) * torch.ceil(10*n_cells / xbar_s)
                n_tiles = torch.ceil(fc_xbars / 64.)

                act_tile_area += LUT_area(n_tiles, m, adc_type)
                tile_area += gate.sum() * LUT_area(n_tiles, m, adc_type)

                # act_tile_latency += LUT_latency(n_tiles, mux, adc, speedup, f)

                # act_tile_area += 668468 * (0.9981761863406945 * n_tiles - 0.041827731986675554)
                # 0.015596502909426158 - 0.041827731987578165
                  # (668468 * (0.9981761863406945 * n_tiles - 0.041827731986675554))
                # tile_latency += gate.sum() * LUT_latency(n_tiles, mux, adc, speedup, f)

                # act_tile_area += 668468 * (0.9981761863406945 * n_tiles - 0.041827731986675554)
                # tile_area += gate.sum()* (668468 * (0.9981761863406945 * n_tiles -0.041827731986675554))

                layers += [nn.AvgPool2d(kernel_size=2, stride=1)]
                features = nn.Sequential(*layers)
                classifier = nn.Linear(int(o_ch.item()), 10)
                network = custom_model(features, classifier)
                # layers += [nn.Linear(int(o_ch.item()), 10)]


                # abs_pe = torch.ceil(torch.ceil(o_ch*2*2/xbar_s)* torch.ceil(1024*4/xbar_s) / n_xbar_tile)
                # total_pe += torch.ceil(o_ch * 2*2 / xbar_s) * torch.ceil(1024 * 4 / xbar_s)
                # abs_tile += abs_pe
                # expected_pe += gate.sum()*abs_pe
                # print(f'-------------- layer {i+1} in_ch {o_ch} pe {fc1.item()} abs_pe {abs_pe.item()} total_pe {total_pe.item()}')

                # print(layers)
                # network = torch.nn.Sequential(*layers)
                # score = score_compute.compute_network_score(network)
                # print(score)
            parallel.append(speedup.item())
            adc_arch.append(adc_type.item())
            mux.append(m.item())
            in_ch = int(o_ch.item())
        print(f' %%%%%% act tile speedup {act_tile_latency.item(), tile_latency.item()} best_latency {best_latency} act_area {act_tile_area.item(), tile_area.item()} adc_arch {np.mean(np.array(adc_arch))} mux {np.mean(np.array(mux))} tiles {np.sum(np.array(t_use))}')
        factor = args.factor
        target_area = torch.FloatTensor([total_area]).cuda()
        # target_util = torch.FloatTensor([1*total_layers]).cuda()

        # mem_util = total_pe / (total_tiles*n_xbar_tile)
        # reg_loss = t_speedup + 10*loss_fn(tile_area, total_area)
        # r_loss_list.append(reg_loss.item())
        # tile_list.append(abs_tile.item())
        # reg_loss += 0.4*loss_fn(expected_util, target_util)
        # error = 0.0001*tile_latency + 0.1*1e-7*(loss_fn(tile_area, total_area))
        # print((loss_fn(tile_area, total_area))*1e-6, tile_latency)
        error = 0.01 * tile_latency + 0.1 * 1e-6 * (loss_fn(tile_area, total_area))
        # error = 0.001*tile_latency*1e-3 + 0.1*(loss_fn(tile_area, total_area))*1e-9

        # print(0.0001*tile_latency, 1*1e-6*torch.sqrt(loss_fn(tile_area, total_area)))
        err_list.append(error.item())
        # pe_in_hardware = float(total_tiles*n_pe_tile)
        if np.abs(act_tile_area.item()-args.t_area) < (args.area_tolerance * args.t_area / 100) and o_list != prev_list and act_tile_latency.item() < best_latency:

            # network = torch.nn.Sequential(*layers)

            # print(network)
            # print('enters here')
            # print(prev_layers==layers)
            # print(layers)
            # for i,_ in enumerate(layers):
            #     if isinstance(layers[i], torch.nn.Conv2d) and len(prev_layers)>1:
            #         print(f'layer {layers[i]}      prev_layer {prev_layers[i]}')

            # for i,_ in enumerate(o_list):
            #     if len(prev_list) > 1:
            #         print(f'o_list {o_list[i]}      prev_list {prev_list[i]}')


            prev_list = o_list
            score = score_compute.compute_network_score(network)
            print(score)
            # if score > best_score and layers != best_network: # and score > 1600 and score <= 1650:
            if score > 1650 and score <= 1700 and layers != best_network:
                # print(f' %%%%%% total latency {total_latency.item()} best_latency {best_latency} best_score {best_score} act_area {act_tile_area.item(), tile_area.item()} parallel {parallel} mux {mux} t_use {t_use}')

                print('hello')
                best_diff = act_tile_latency.item()
                best_area = act_tile_area.detach().cpu().numpy()
                best_latency = act_tile_latency.detach().cpu().numpy()
                best_mux = mux
                best_network = layers
                best_par = parallel
                best_tile_use = t_use
                best_score = score
                best_adc_arch = adc_arch
                best_adc_mean = np.mean(np.array(best_adc_arch))
                best_mux_mean = np.mean(np.array(best_mux))
                best_tile_sum = np.sum(np.array(best_tile_use))
                # best_sup = sup
                # best_util = util_list
                # best_mem_util = mem_util
                best_arch = copy.deepcopy(arch_params)
            # exp_pe = expected_pe

    flag=4
    optimizer.zero_grad()
    error.backward(retain_graph=True)
    optimizer.step()
arch = []
for i in range(total_layers):
    prob = s(best_arch[i])
    index = torch.argmax(prob)
    a = network_arch[i][index]  # [1*(prob == torch.max(prob)) == 1]
    print(a)
    arch.append(a)

arch2 = []
arch3 = []
csv = []
in_ch = 3
l_c = 0
for i in arch:
    if i[2] == 3:
        pad = 1
    if i[2] == 5:
        pad = 2
    if i[2] == 7:
        pad = 3
    if int(i[3].item()) == 4 or int(i[3].item()) == 2:
        f_size = 6
    else:
        f_size = int(i[3].item())

    csv.append((f_size+2, f_size+2, int(in_ch),  int(i[2].item()),
                int(i[2].item()), int(i[1].item()), 0, 1, int(i[5]), int(i[0]) ))

    if int(in_ch) == 64 and int(i[1].item()) == 512:
        csv.append((int(i[3].item())+2, int(i[3].item())+2,512,3,3,64,0,1,int(i[5]), int(i[0])))
    else:
        csv.append((int(i[3].item())+2, int(i[3].item())+2,64,3,3,512,0,1,int(i[5]), int(i[0])))

    arch3.append(('C', int(i[1].item()), int(i[2].item()), 'same', int(8)))
    if l_c == 1 or l_c == 3 or l_c == 6 or l_c == 9:
        arch3.append(('M',2,2))


    arch2.append(('C', int(in_ch), int(i[1].item()), int(i[2].item()), 'same', int(8)))
    if int(in_ch) == 64 and int(i[1].item()) == 512:
        arch2.append(('C', 512, 64, 3, 'same', int(8)))
    else:
        arch2.append(('C',64, 512, 3, 'same', int(8)))
    if l_c == 1 or l_c == 3 or l_c == 6 or l_c == 9:
        arch2.append(('M',2,2))
    l_c += 1
    in_ch = int(i[1].item())

print(f'latency {best_latency.item()},\nbest_adc_arch {best_adc_arch}, \nbest_area {best_area}, \nspeedup {best_par}, \ncolumn_sharing {best_mux} \nbest_t_use {best_tile_use} \nbest_score {best_score} \n adc_mean {best_adc_mean} mux_mean {best_mux_mean} tile_sum {best_tile_sum}')
# torch.save(arch_param_epoch, './cifar10/arch_param_epoch.pt')
# torch.save(lat_err_list, './lat_err_list.pt')
# torch.save(err_list, './err_list.pt')
# torch.save(r_loss_list, './r_loss_list.pt')
# torch.save(tile_list, './tile_list.pt')
for i in range(len(csv)):
    print(str(csv[i])[1:-1])

# print(csv)
print(arch3)
print(arch2)
print('\n')
