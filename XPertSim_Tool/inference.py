import argparse
import os
import time
from utee import misc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from utee import wage_util
from models import dataset
import torchvision.models as models
from utee import hook
#from IPython import embed
from datetime import datetime
from subprocess import call
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--dataset', default='cifar10', help='cifar10|cifar100|imagenet')
parser.add_argument('--model', default='custom', help='VGG8|DenseNet40|ResNet18')
parser.add_argument('--mode', default='WAGE', help='WAGE|FP')
parser.add_argument('--batch_size', type=int, default=500, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1e-3)')
parser.add_argument('--decreasing_lr', default='140,180', help='decreasing strategy')
parser.add_argument('--wl_weight', type=int, default=8)
parser.add_argument('--wl_grad', type=int, default=8)
parser.add_argument('--wl_activate', type=int, default=4)
parser.add_argument('--wl_error', type=int, default=8)
# Hardware Properties
# if do not consider hardware effects, set inference=0
parser.add_argument('--inference', type=int, default=0, help='run hardware inference simulation')
parser.add_argument('--subArray', type=int, default=128, help='size of subArray (e.g. 128*128)')
parser.add_argument('--ADCprecision', type=int, default=6, help='ADC precision (e.g. 5-bit)')
parser.add_argument('--cellBit', type=int, default=8, help='cell precision (e.g. 4-bit/cell)')
parser.add_argument('--onoffratio', type=float, default=1000, help='device on/off ratio (e.g. Gmax/Gmin = 3)')
# if do not run the device retention / conductance variation effects, set vari=0, v=0
parser.add_argument('--vari', type=float, default=0, help='conductance variation (e.g. 0.1 standard deviation to generate random variation)')
parser.add_argument('--t', type=float, default=0, help='retention time')
parser.add_argument('--v', type=float, default=0, help='drift coefficient')
parser.add_argument('--detect', type=int, default=0, help='if 1, fixed-direction drift, if 0, random drift')
parser.add_argument('--target', type=float, default=0, help='drift target for fixed-direction drift, range 0-1')
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()

args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs','gpu','ngpu','debug'])

misc.logger.init(args.logdir, 'test_log' + current_time)
logger = misc.logger.info

misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
	logger('{}: {}'.format(k, v))
logger("========================================")

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

# data loader and model
assert args.dataset in ['cifar10', 'cifar100', 'tinyimagenet'], args.dataset
if args.dataset == 'cifar10':
    train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1)
elif args.dataset == 'cifar100':
    train_loader, test_loader = dataset.get_cifar100(batch_size=args.batch_size, num_workers=1)
elif args.dataset == 'tinyimagenet':
    train_loader, test_loader = dataset.get_tiny(batch_size=args.batch_size, num_workers=1)
else:
    raise ValueError("Unknown dataset type")
    
assert args.model in ['VGG8', 'DenseNet40', 'ResNet18', 'custom'], args.model
# if args.model == 'custom':
#     from models import VGG
#     # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=WAGE/model=VGG8/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/best-0.pth'  #Model 1
#     # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.001/mode=WAGE/model=VGG8/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/best-0.pth'
#     # model_path= './custom_model_pths/vgg16_bn_91.pth'
#     # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=WAGE/model=VGG8/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/best-0.pth'
#     # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=WAGE/model=VGG8/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/best-0.pth'  #10 layers
#     # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=WAGE/model=VGG8/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/best-0.pth' #./log/VGG8.pth'   # WAGE mode pretrained model
#     modelCF = VGG.vgg8(args = args, logger=logger, pretrained = None)
if args.model == 'custom':
    from models import VGG

    model_path = '../../cifar10_train/pytorch-cifar/checkpoint/ckpt.pth'
    # model_path = 'custom_model_pths/Model_5_FP_90.pth'
    # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.001/mode=FP/model=VGG8/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/best-37.pth'
    modelCF = VGG.vgg8(pretrained=None, args=args, logger=logger)
    modelCF = torch.nn.DataParallel(modelCF)
    # modelCF.load_state_dict(torch.load(model_path)['net'])
    print(modelCF)

    # from models import VGG
    #
    # modelCF = VGG.vgg8(args=args, logger=logger, pretrained=None)
    # modelCF = torch.nn.DataParallel(modelCF)
    # # modelCF.load_state_dict(torch.load(model_path))
    # print(modelCF)

    # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=WAGE/model=VGG8/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/best-0.pth'  #Model 1
    # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.001/mode=WAGE/model=VGG8/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/best-0.pth'
    # model_path= './custom_model_pths/vgg16_bn_91.pth'
    # model_path = './ckpt.pth'
    # model_path = './custom_model_pths/Custom_2_debug_bna_mdacadc_8.pth'
    # modelCF = software_vgg.VGG_soft('Custom_2')

    # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=WAGE/model=VGG8/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/best-0.pth'
    # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=WAGE/model=VGG8/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/best-0.pth'  #10 layers
    # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=WAGE/model=VGG8/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/best-0.pth' #./log/VGG8.pth'   # WAGE mode pretrained model


elif args.model == 'DenseNet40':
    from models import DenseNet
    # model_path = './log/DenseNet40.pth'     # WAGE mode pretrained model
    modelCF = DenseNet.densenet40(args = args, logger=logger, pretrained = None)
elif args.model == 'ResNet18':
    from models import ResNet_CIFAR10
    # FP mode pretrained model, loaded from 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    # model_path = './log/xxx.pth'
    # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=256/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.1/mode=FP/model=ResNet18/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/fp_resnet18.pth'
    # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=WAGE/model=ResNet18/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/best-0.pth'
    # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=WAGE/model=ResNet18/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/latest.pth'
    # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=256/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=FP/model=ResNet18/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/Resnet_model1.pth'
    # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=WAGE/model=ResNet18/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/hw_eval.pth'
    # model_path = '/gpfs/loomis/project/panda/am3554/DNN_NeuroSim_V1.3/Inference_pytorch/log/default/ADCprecision=5/batch_size=256/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.0001/mode=WAGE/model=ResNet18/onoffratio=10/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=8/best-4.pth'
    modelCF = ResNet_CIFAR10.resnet18(args = args, logger=logger, pretrained = None)
    print('model_inst')
    print(modelCF)
else:
    raise ValueError("Unknown model type")

if args.cuda:
	modelCF.cuda()

best_acc, old_file = 0, None
t_begin = time.time()
# ready to go
modelCF.eval()

test_loss = 0
correct = 0
trained_with_quantization = True

criterion = torch.nn.CrossEntropyLoss()
# criterion = wage_util.SSE()

# for data, target in test_loader:
for i, (data, target) in enumerate(test_loader):
    print(f'batch {i}')
    if i==0:
        hook_handle_list = hook.hardware_evaluation(modelCF,args.wl_weight,args.wl_activate,args.model,args.mode)
    indx_target = target.clone()
    if args.cuda:
        # if args.dataset == 'cifar10':
      data, target = data.cuda(), target.cuda()
        # else:
        # data = torch.rand(size=[16,3,64,64]).cuda()
    with torch.no_grad():
        # data = torch.rand(size=[16, 3, 64, 64]).cuda()
        # if args.dataset != 'cifar10':
        #     data = torch.rand(size=[16,3,64,64]).cuda()
        data, target = Variable(data), Variable(target)
        output = modelCF(data)
        # output = modelCF(inp, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], i,
        #                args.model)

        # test_loss_i = criterion(output, target)
        # test_loss += test_loss_i.data
        # pred = output.data.max(1)[1]  # get the index of the max log-probability
        # correct += pred.cpu().eq(indx_target).sum()
    if i==0:
        hook.remove_hook_list(hook_handle_list)

# test_loss = test_loss / len(test_loader)  # average over number of mini-batch
# acc = 100. * correct / len(test_loader.dataset)
#
# accuracy = acc.cpu().data.numpy()

if args.inference:
    print(" --- Hardware Properties --- ")
    print("subArray size: ")
    print(args.subArray)
    print("ADC precision: ")
    print(args.ADCprecision)
    print("cell precision: ")
    print(args.cellBit)
    print("on/off ratio: ")
    print(args.onoffratio)
    print("variation: ")
    print(args.vari)

# logger('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
# 	test_loss, correct, len(test_loader.dataset), acc))

call(["/bin/bash", './layer_record_'+str(args.model)+'/trace_command.sh'])
