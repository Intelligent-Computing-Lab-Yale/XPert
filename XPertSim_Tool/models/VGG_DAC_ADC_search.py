from utee import misc
# print = misc.logger.info
import torch.nn as nn
from modules.quantization_cpu_np_infer_DAC_ADC import QConv2d,  QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import torch


class masking(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a_param):

        m = nn.Softmax()
        p_vals = m(a_param)

        # print(p_vals)

        gates = torch.FloatTensor([1]).cuda() * (1. *(p_vals == torch.max(p_vals))).float()
        # print(gates)
        # print(type(gates))
        # print(gates)

        # mixed_op = input[gates == 1.]
        # if (mixed_op.size(0) == 1):
        #     mixed_op = mixed_op[0]
        # else:
        #     perm = torch.randperm(mixed_op.size(0))
        #     idx = perm[:1]
        #     mixed_op = mixed_op[idx]
        #
        # # print(mixed_op.size())
        #
        # ctx.save_for_backward(input, a_param, p_vals, gates)

        return gates

    @staticmethod
    def backward(ctx, grad_output):
        # print(grad_output)
        # global arch_grads
        # global flag
        #
        # input, a_param, p_vals, gates = ctx.saved_tensors
        #
        # grad_input = grad_output.clone()
        #
        # additional = torch.zeros_like(a_param).cuda()
        #
        # delta = torch.eye(8).cuda()
        #
        # binary_grads = torch.zeros_like(gates.data).cuda()
        #
        # # print(input.size())
        # binary_grads = torch.sum(grad_output * input, dim=(1, 2, 3, 4))
        #
        # for i in range(8):
        #     for j in range(8):
        #         additional[i] = additional[i] + p_vals[j] * (delta[i, j] - p_vals[i]) * binary_grads[j]
        # arch_grads[flag] = additional
        #
        # flag = flag - 1
        #
        # grad_return = []
        #
        # for k in range(4):
        #     grad_return.append(grad_input * gates[k])
        #
        # grad_return = torch.stack(grad_return).cuda()
        grad_return = grad_output
        return grad_return, None

class VGG(nn.Module):
    def __init__(self, args, features, num_classes):
        super(VGG, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = make_layers([('L', 2048, 1024),
                                       ('L', 1024, num_classes)],
                                       args)
        self.s = nn.Softmax()
        # print(self.features)
        # print(self.classifier)

    def forward(self, x, network_adc_nodes, arch_params):
        count = 0
        # print(highest_adc_params , adc_prec)
        prob_vals = []
        adc_prec, dac_prec = [], []
        for i in self.features:
            if isinstance(i, QConv2d):
                # print(highest_adc_params[count])
                # print(i(x, adc_prec[count]))
                # print(highest_adc_params[0,count].grad_fn)
                # print(highest_adc_params[0,count].requires_grad)
                # print(f'arch params arch_params[0,count]')
                # print(network_adc_nodes.size())
                # print(arch_params)

                # if prob[index]*2
                # x = prob[index]*(1/(prob[index]-0.01)) * i(x, network_adc_nodes[count, index])
                # if training:
                prob = self.s(arch_params[count])
                index = torch.argmax(prob)

                gate = masking.apply(arch_params[count])

                x = gate.sum() * i(x, network_adc_nodes[count, index][1].item(), network_adc_nodes[count, index][0].item())
                # print(f' layer {count} adc_prec {network_adc_nodes[count, index]}  prob {prob[index]} {prob[index].device}')
                adc_prec.append(int(network_adc_nodes[count, index][0].item()))
                dac_prec.append(int(network_adc_nodes[count, index][1].item()))
                prob_vals.append(prob[index].item())

                # if prob[index] <= 0.1:
                #     x = prob[index] * 10 * i(x, network_adc_nodes[count, index])
                # elif prob[index] > 0.1 and prob[index] <= 0.2:
                #     x = prob[index] * 5 * i(x, network_adc_nodes[count, index])
                # elif prob[index] > 0.2 and prob[index] <= 0.3:
                #     x = prob[index] * 3.33 * i(x, network_adc_nodes[count, index])
                # elif prob[index] > 0.3 and prob[index] <= 0.4:
                #     x = prob[index] * 2.5 * i(x, network_adc_nodes[count, index])
                # elif prob[index] > 0.4 and prob[index] <= 0.5:
                #     x = prob[index] * 2 * i(x, network_adc_nodes[count, index])
                # elif prob[index] > 0.5 and prob[index] <= 0.6:
                #     x = prob[index] * 1.66 * i(x, network_adc_nodes[count, index])
                # elif prob[index] > 0.6 and prob[index] <= 0.7:
                #     x = prob[index] * 1.42 * i(x, network_adc_nodes[count, index])
                # elif prob[index] > 0.7 and prob[index] <= 0.8:
                #     x = prob[index] * 1.25 * i(x, network_adc_nodes[count, index])
                # elif prob[index] > 0.8 and prob[index] <= 0.9:
                #     x = prob[index] * 1.11 * i(x, network_adc_nodes[count, index])
                # elif prob[index] > 0.9 and prob[index] <= 1:
                #     x = prob[index] * 1 * i(x, network_adc_nodes[count, index])

                # if prob[index] < 0.9:
                #     x = prob[index] * 2.7 * i(x, network_adc_nodes[count, index])
                # else:
                #     x = prob[index] * i(x, network_adc_nodes[count, index])
                # print(prob[index])

                count += 1
                # print('this')
            else:
                x = i(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, adc_prec, dac_prec, prob_vals


def make_layers(cfg, args ):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        if v[0] == 'C':
            out_channels = v[1]
            if v[3] == 'same':
                padding = v[2]//2
            else:
                padding = 0
            if args.mode == "WAGE":
                conv2d = QConv2d(in_channels, out_channels, kernel_size=v[2], padding=padding,
                                 wl_input = args.wl_activate,wl_activate=args.wl_activate,
                                 wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                 subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                 name = 'Conv'+str(i)+'_', model = args.model)
            elif args.mode == "FP":
                conv2d = FConv2d(in_channels, out_channels, kernel_size=v[2], padding=padding,
                                 wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                 subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                 name = 'Conv'+str(i)+'_' )
            non_linearity_activation =  nn.ReLU()
            bnorm = torch.nn.BatchNorm2d(out_channels)
            layers += [conv2d, bnorm, non_linearity_activation]
            in_channels = out_channels
        if v[0] == 'L':
            # if args.mode == "WAGE":
            #     linear = QLinear(in_features=v[1], out_features=v[2],
            #                     logger=logger, wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
            #                     wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
            #                     subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
            #                     name='FC'+str(i)+'_', model = args.model)
            # elif args.mode == "FP":
            linear = FLinear(in_features=v[1], out_features=v[2],
                             wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                             subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                             name='FC'+str(i)+'_')
            bnorm = torch.nn.BatchNorm1d(v[2])
            if i < len(cfg)-1:
                non_linearity_activation =  nn.ReLU()
                layers += [linear, bnorm, non_linearity_activation]
            else:
                layers += [linear]
    return nn.Sequential(*layers)



cfg_list = {
    'vgg8': [('C', 128, 3, 'same', 2.0),
                ('C', 128, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('M', 2, 2)],

    'vgg16': [('C', 64, 3, 'same', 2.0),
                ('C', 64, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 128, 3, 'same', 16.0),
                ('C', 128, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 32.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('C', 512, 3, 'same', 32.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('C', 512, 3, 'same', 32.0)
              ],

    'custom_1000' : [('C', 64, 5, 'same', 8), ('M', 2, 2), ('C', 128, 3, 'same', 8), ('C', 64, 3, 'same', 8), ('C', 128, 3, 'same', 8),
                ('C', 64, 3, 'same', 8), ('C', 64, 3, 'same', 8), ('C', 64, 3, 'same', 8), ('C', 128, 5, 'same', 8),
                ('C', 64, 3, 'same', 8), ('C', 128, 5, 'same', 8)],

    'custom_1000_5' : [(128, 5, 2, 32.0, 8.0), (64, 5, 2, 16.0, 16.0), (64, 5, 2, 16.0, 64.0), (128, 3, 1, 16.0, 64.0), (64, 3, 1, 8.0, 64.0)],

    'custom_1000_35' : [('C', 128, 5, 'same', 8), ('M', 2, 2), ('C', 64, 5, 'same', 8), ('C', 64, 5, 'same', 8), ('C', 128, 3, 'same', 8), ('M', 2, 2),
                            ('C', 64, 3, 'same', 8)],

    'custom_8000_35_6' : [('C', 64, 7, 'same', 8), ('M', 2, 2), ('C', 128, 3, 'same', 8), ('C', 64, 3, 'same', 8), ('C', 64, 3, 'same', 8), ('M', 2, 2), ('C', 64, 5, 'same', 8), ('C', 64, 3, 'same', 8)],

    'custom_10000_200_6+fc' : [('C', 128, 7, 'same', 8), ('M', 2, 2), ('C', 64, 5, 'same', 8), ('C', 128, 7, 'same', 8), ('C', 64, 5, 'same', 8), ('M', 2, 2), ('C', 64, 5, 'same', 8), ('C', 64, 5, 'same', 8)],

    'Model_1': [('C', 16, 3, 'same', 8), ('M', 2, 2), ('C', 64, 7, 'same', 8), ('C', 32, 7, 'same', 8),
                ('C', 32, 5, 'same', 8), ('C', 64, 5, 'same', 8), ('C', 32, 7, 'same', 8), ('C', 32, 7, 'same', 8),
                ('C', 128, 3, 'same', 8), ('C', 64, 5, 'same', 8), ('C', 128, 3, 'same', 8)],

    'Model_2': [('C', 128, 7, 'same', 8), ('M', 2, 2), ('C', 64, 7, 'same', 8), ('C', 512, 3, 'same', 8),
                    ('C', 64, 7, 'same', 8), ('M', 2, 2), ('C', 64, 7, 'same', 8), ('C', 64, 5, 'same', 8),
                    ('C', 256, 7, 'same', 8), ('M', 2, 2), ('C', 128, 3, 'same', 8)],

    'Model_4': [('C', 128, 3, 'same', 8), ('M', 2, 2), ('C', 128, 5, 'same', 8), ('C', 128, 5, 'same', 8),
                ('C', 256, 3, 'same', 8), ('M', 2, 2),
                ('C', 512, 5, 'same', 8), ('C', 128, 5, 'same', 8),
                ('C', 512, 5, 'same', 8), ('C', 64, 5, 'same', 8), ('M', 2, 2), ('C', 64, 5, 'same', 8),
                ('C', 256, 5, 'same', 8), ('C', 64, 5, 'same', 8), ('C', 128, 3, 'same', 8), ('M', 2, 2)],

    'Model_2_3': [('C', 128, 3, 'same', 8), ('M', 2, 2), ('C', 64, 3, 'same', 8), ('C', 512, 3, 'same', 8),
                ('C', 64, 3, 'same', 8), ('M', 2, 2), ('C', 64, 3, 'same', 8), ('C', 64, 3, 'same', 8),
                ('C', 256, 3, 'same', 8), ('M', 2, 2), ('C', 128, 3, 'same', 8)],

    'Model_5' : [('C', 64, 3, 'same', 8), ('M', 2, 2), ('C', 256, 3, 'same', 8), ('C', 128, 3, 'same', 8), ('C', 64, 5, 'same', 8), ('M', 2, 2), ('C', 64, 5, 'same', 8), ('C', 512, 5, 'same', 8),
                     ('C', 64, 5, 'same', 8), ('C', 256, 5, 'same', 8), ('M', 2, 2), ('C', 512, 3, 'same', 8), ('C', 256, 5, 'same', 8), ('C', 64, 5, 'same', 8), ('C', 512, 5, 'same', 8), ('M', 2, 2)],


    'Model_5_3' : [('C', 64, 3, 'same', 8), ('M', 2, 2), ('C', 256, 3, 'same', 8), ('C', 128, 3, 'same', 8), ('C', 64, 3, 'same', 8), ('M', 2, 2), ('C', 64, 3, 'same', 8), ('C', 512, 3, 'same', 8),
                 ('C', 64, 3, 'same', 8), ('C', 256, 3, 'same', 8), ('M', 2, 2), ('C', 512, 3, 'same', 8), ('C', 256, 3, 'same', 8), ('C', 64, 3, 'same', 8), ('C', 512, 3, 'same', 8), ('M', 2, 2)],


}

def vgg8( args, pretrained=None):
    cfg = cfg_list['Model_5']
    layers = make_layers(cfg, args)
    model = VGG(args,layers, num_classes=10)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


