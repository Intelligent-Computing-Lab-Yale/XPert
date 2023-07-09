import torch
import torch.nn as nn
import torch.nn.functional as F
from utee import wage_initializer, wage_quantizer
from torch._jit_internal import weak_script_method
import numpy as np
import copy

class BinOp():
    def __init__(self, model):
        # count the number of Conv2d
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1
        print(count_Conv2d)
        start_range = 1
        end_range = count_Conv2d
        self.bin_range = np.linspace(start_range,
                                     end_range, end_range - start_range + 1) \
            .astype('int').tolist()
        print(self.bin_range)
        # kbit_conn = numpy.array([0, 1, 2, 4, 6, 7, 8, 9, 11, 12,13, 14, 16, 17, 18, 19]) #layers whose weights need to be made k-bit
        # kbit_conn = kbit_conn.astype('int').tolist()
        # print(kbit_conn)
        # raw_input()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        print(self.num_of_params)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                print(m)
                index = index + 1
                # if index in self.bin_range:
                #    tmp = m.weight.data.clone()
                #    self.saved_params.append(tmp)
                #    self.target_modules.append(m.weight)
                # if index in kbit_conn:
                print('Making k-bit')  # Know which layers weights are being made k-bit
                # raw_input()
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp)
                self.target_modules.append(m.weight)

    def binarization(self):
        # self.meancenterConvParams()
        # self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            # print(index)
            # print(s)
            negMean = self.target_modules[index].data.mean(1, keepdim=True). \
                mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        # kbit_conn = numpy.array([0, 1, 3, 4, 6, 7, 8, 9, 11, 12,13, 14, 16, 17, 18, 19])
        # num_bits = numpy.array([65408  ,61440  ,63488  ,63488  ,57344  ,61440  ,57344  ,57344  ,61440  ,49152  ,61440  ,61440
        #                        ,64512 ,65408  ,65280  ,65408  ,65280  ,65024  ,57344  ,49152  ,65408])/257#numpy.array([255,  192,    224,    224,    128,    192,    192,    192,    192,    128,    192,    192,    224,    240,    240,    248,    240,    240,    128,    128,    255])
        # num_bits = numpy.array([65408  ,255    ,255    ,255    ,255    ,255 ,255   ,255    ,255    ,255    ,255,255    ,65408  ,65280  ,65408  ,65280  ,65024  ,255    ,49152  ,65408])
        num_bits = weight_conn
        # kbit_conn = kbit_conn.astype('int').tolist()
        for index in range(self.num_of_params):
            # n = self.target_modules[index].data[0].nelement()
            # s = self.target_modules[index].data.size()
            # m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
            #        .sum(2, keepdim=True).sum(1, keepdim=True).div(n)

            # if index in kbit_conn:
            # k-bit weights
            x = self.target_modules[index].data
            xmax = x.abs().max()
            v0 = 1
            v1 = 2
            v2 = -0.5
            y = num_bits[index]  # +std[index]#torch.normal(0,std[index], size=(1,1))#2.**num_bits[index] - 1.
            # print(y)
            x = x.add(v0).div(v1)
            # print(x)
            x = x.mul(y).round_()
            x = x.div(y)
            x = x.add(v2)
            x = x.mul(v1)
            n_bits = 4
            W_sbits = torch.round(x * 2 ** (n_bits - 1))
            W_sbits = W_sbits / 2 ** (n_bits - 1)
            # if index == 0:
            #     print('saving weights')
            #     torch.save(W_sbits, './sw_outputs/sw_q_wt1')
            self.target_modules[index].data = W_sbits

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True) \
                .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0
            m[weight.gt(1.0)] = 0
            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad.data = \
            #         self.target_modules[index].grad.data.mul(m)
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True) \
                .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0 - 1.0 / s[1]).mul(n)
def quantize(x, n_bits):
    v0 = 1
    v1 = 2
    v2 = -0.5
    y = n_bits  # +std[index]#torch.normal(0,std[index], size=(1,1))#2.**num_bits[index] - 1.
    # print(y)
    x = x.add(v0).div(v1)
    # print(x)
    x = x.mul(y).round_()
    x = x.div(y)
    x = x.add(v2)
    x = x.mul(v1)
    W_sbits = torch.round(x * 2 ** (n_bits - 1))
    W_sbits = W_sbits / 2 ** (n_bits - 1)

    return  W_sbits

class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, logger=None, clip_weight=False, wage_init=False,
                 quantize_weight=False, clip_output=False, quantize_output=False,
                 wl_input=8, wl_activate=8, wl_error=8, wl_weight=8, inference=0, onoffratio=10, cellBit=1,
                 subArray=128, ADCprecision=5, vari=0, t=0, v=0, detect=0, target=0, debug=0, name='Qconv', model=None):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_input = wl_input
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.name = name
        self.model = model
        # self.binop = BinOp(self)
        self.scale = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)

    @weak_script_method
    def forward(self, input, dac, adc):
        self.wl_input = dac
        self.ADCprecision = adc
        # input = wage_quantizer.Q(input, self.wl_input)
        # self.wl_activate = dac
        # weight = quantize(self.weight, self.wl_weight)
        # weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        # weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
        # outputOrignal = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)
        # print(weight.shape[2], weight.shape[3])
        # print(self.model, self.inference)
        if self.inference == 1 and self.model == 'vgg':
            # set parameters for Hardware Inference
            onoffratio = self.onoffratio
            upper = 1
            lower = 1 / onoffratio

            output = torch.zeros_like(outputOrignal)
            del outputOrignal
            cellRange = 2 ** self.cellBit  # cell precision is 4

            # Now consider on/off ratio
            dummyP = torch.zeros_like(weight)
            dummyP[:, :, :, :] = (cellRange - 1) * (upper + lower) / 2

            for i in range(self.weight.shape[2]):
                for j in range(self.weight.shape[3]):
                    # need to divide to different subArray
                    numSubArray = int(weight.shape[1] / self.subArray)
                    # cut into different subArrays
                    if numSubArray == 0:
                        mask = torch.zeros_like(weight)
                        mask[:, :, i, j] = 1
                        if weight.shape[1] == 3:
                            # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                            X_decimal = torch.round((2 ** bitWeight - 1) / 2 * (weight + 1) + 0) * mask
                            outputP = torch.zeros_like(output)
                            outputD = torch.zeros_like(output)
                            for k in range(int(bitWeight / self.cellBit)):
                                remainder = torch.fmod(X_decimal, cellRange) * mask
                                # retention
                                remainder = wage_quantizer.Retention(remainder, self.t, self.v, self.detect,
                                                                     self.target)
                                X_decimal = torch.round((X_decimal - remainder) / cellRange) * mask
                                # Now also consider weight has on/off ratio effects
                                # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                                remainderQ = (upper - lower) * (remainder - 0) + (
                                            cellRange - 1) * lower  # weight cannot map to 0, but to Gmin
                                # remainderQ = remainderQ + remainderQ*torch.normal(0., torch.full(remainderQ.size(),self.vari, device='cuda'))
                                outputPartial = F.conv2d(input, remainderQ * mask, self.bias, self.stride, self.padding,
                                                         self.dilation, self.groups)
                                outputDummyPartial = F.conv2d(input, dummyP * mask, self.bias, self.stride,
                                                              self.padding, self.dilation, self.groups)
                                scaler = cellRange ** k
                                outputP = outputP + outputPartial * scaler * 2 / (1 - 1 / onoffratio)
                                outputD = outputD + outputDummyPartial * scaler * 2 / (1 - 1 / onoffratio)
                            outputP = outputP - outputD
                            output = output + outputP
                        else:
                            # quantize input into binary sequence
                            inputQ = torch.round((2 ** bitActivation - 1) / 1 * (input - 0) + 0)
                            outputIN = torch.zeros_like(output)
                            for z in range(bitActivation):
                                inputB = torch.fmod(inputQ, 2)
                                inputQ = torch.round((inputQ - inputB) / 2)
                                outputP = torch.zeros_like(output)
                                # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                                X_decimal = torch.round((2 ** bitWeight - 1) / 2 * (weight + 1) + 0) * mask
                                outputD = torch.zeros_like(output)
                                for k in range(int(bitWeight / self.cellBit)):
                                    remainder = torch.fmod(X_decimal, cellRange) * mask
                                    # retention
                                    remainder = wage_quantizer.Retention(remainder, self.t, self.v, self.detect,
                                                                         self.target)
                                    X_decimal = torch.round((X_decimal - remainder) / cellRange) * mask
                                    # Now also consider weight has on/off ratio effects
                                    # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                    # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                                    remainderQ = (upper - lower) * (remainder - 0) + (
                                                cellRange - 1) * lower  # weight cannot map to 0, but to Gmin
                                    # remainderQ = remainderQ + remainderQ*torch.normal(0., torch.full(remainderQ.size(),self.vari, device='cuda'))
                                    outputPartial = F.conv2d(inputB, remainderQ * mask, self.bias, self.stride,
                                                             self.padding, self.dilation, self.groups)
                                    outputDummyPartial = F.conv2d(inputB, dummyP * mask, self.bias, self.stride,
                                                                  self.padding, self.dilation, self.groups)
                                    # Add ADC quanization effects here !!!
                                    outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                                    outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial,
                                                                                           self.ADCprecision)
                                    scaler = cellRange ** k
                                    outputP = outputP + outputPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                                    outputD = outputD + outputDummyPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                                scalerIN = 2 ** z
                                outputIN = outputIN + (outputP - outputD) * scalerIN
                            output = output + outputIN / (2 ** bitActivation)
                    else:
                        # quantize input into binary sequence
                        inputQ = torch.round((2 ** bitActivation - 1) / 1 * (input - 0) + 0)
                        outputIN = torch.zeros_like(output)
                        for z in range(bitActivation):
                            inputB = torch.fmod(inputQ, 2)
                            inputQ = torch.round((inputQ - inputB) / 2)
                            outputP = torch.zeros_like(output)
                            for s in range(numSubArray):
                                mask = torch.zeros_like(weight)
                                mask[:, (s * self.subArray):(s + 1) * self.subArray, i, j] = 1
                                # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                                X_decimal = torch.round((2 ** bitWeight - 1) / 2 * (weight + 1) + 0) * mask
                                outputSP = torch.zeros_like(output)
                                outputD = torch.zeros_like(output)
                                for k in range(int(bitWeight / self.cellBit)):
                                    remainder = torch.fmod(X_decimal, cellRange) * mask
                                    # retention
                                    remainder = wage_quantizer.Retention(remainder, self.t, self.v, self.detect,
                                                                         self.target)
                                    X_decimal = torch.round((X_decimal - remainder) / cellRange) * mask
                                    # Now also consider weight has on/off ratio effects
                                    # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                    # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
                                    remainderQ = (upper - lower) * (remainder - 0) + (
                                                cellRange - 1) * lower  # weight cannot map to 0, but to Gmin
                                    # remainderQ = remainderQ + remainderQ*torch.normal(0., torch.full(remainderQ.size(),self.vari, device='cuda'))
                                    outputPartial = F.conv2d(inputB, remainderQ * mask, self.bias, self.stride,
                                                             self.padding, self.dilation, self.groups)
                                    outputDummyPartial = F.conv2d(inputB, dummyP * mask, self.bias, self.stride,
                                                                  self.padding, self.dilation, self.groups)
                                    # Add ADC quanization effects here !!!
                                    outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                                    outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial,
                                                                                           self.ADCprecision)
                                    scaler = cellRange ** k
                                    outputSP = outputSP + outputPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                                    outputD = outputD + outputDummyPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                                # !!! Important !!! the dummy need to be multiplied by a ratio
                                outputSP = outputSP - outputD  # minus dummy column
                                outputP = outputP + outputSP
                            scalerIN = 2 ** z
                            outputIN = outputIN + outputP * scalerIN
                        output = output + outputIN / (2 ** bitActivation)
            output = output / (2 ** bitWeight)  # since weight range was convert from [-1, 1] to [-256, 256]
        # elif self.inference == 1:
        #     weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        #     weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
        #     weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
        #     input = wage_quantizer.Q(input,self.wl_input)
        #     output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        #     output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        elif self.inference == 1 and self.model=='custom':
            print('hello Im tere')
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
            weight = wage_quantizer.Retention(weight, self.t, self.v, self.detect, self.target)
            # weight = wage_quantizer.Q(self.weight, self.wl_weight).detach()
            weight = weight + weight * torch.normal(0., torch.full(weight.size(), self.vari, device='cuda'))

            input = wage_quantizer.Q(input, self.wl_input)
            output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        else:
            print('hello Im also tere')
            # original WAGE QCov2d
            # print('duh hello')
            # weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            # weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
            # weight = wage_quantizer.Retention(weight, self.t, self.v, self.detect, self.target)
            # input = wage_quantizer.Q(input, self.wl_input)
            # output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            # output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)

            weight = quantize(self.weight, self.wl_weight)
            # weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            # weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
            # weight = wage_quantizer.Retention(weight, self.t, self.v, self.detect, self.target)
            output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # output = output / self.scale
        output = wage_quantizer.WAGEQuantizer_f(output, self.wl_activate, self.wl_error)
        # intermed_output = copy.deepcopy(output)
        return output


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, logger=None, clip_weight=False, wage_init=False,
                 quantize_weight=False, clip_output=False, quantize_output=False,
                 wl_input=8, wl_activate=8, wl_error=8, wl_weight=8, inference=0, onoffratio=10, cellBit=1,
                 subArray=128, ADCprecision=5, vari=0, t=0, v=0, detect=0, target=0, debug=0, name='Qlinear',
                 model=None):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_input = wl_input
        self.wl_error = wl_error
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.name = name
        self.model = model
        self.scale = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)

    @weak_script_method
    def forward(self, input):

        weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
        outputOrignal = F.linear(input, weight, self.bias)
        output = torch.zeros_like(outputOrignal)

        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        if self.inference == 1 and self.model == 'VGG8':
            # set parameters for Hardware Inference
            onoffratio = self.onoffratio
            upper = 1
            lower = 1 / onoffratio
            output = torch.zeros_like(outputOrignal)
            cellRange = 2 ** self.cellBit  # cell precision is 4
            # Now consider on/off ratio
            dummyP = torch.zeros_like(weight)
            dummyP[:, :] = (cellRange - 1) * (upper + lower) / 2
            # need to divide to different subArray
            numSubArray = int(weight.shape[1] / self.subArray)

            if numSubArray == 0:
                mask = torch.zeros_like(weight)
                mask[:, :] = 1
                # quantize input into binary sequence
                inputQ = torch.round((2 ** bitActivation - 1) / 1 * (input - 0) + 0)
                outputIN = torch.zeros_like(outputOrignal)
                for z in range(bitActivation):
                    inputB = torch.fmod(inputQ, 2)
                    inputQ = torch.round((inputQ - inputB) / 2)
                    # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                    X_decimal = torch.round((2 ** bitWeight - 1) / 2 * (weight + 1) + 0) * mask
                    outputP = torch.zeros_like(outputOrignal)
                    outputD = torch.zeros_like(outputOrignal)
                    for k in range(int(bitWeight / self.cellBit)):
                        remainder = torch.fmod(X_decimal, cellRange) * mask
                        # retention
                        remainder = wage_quantizer.Retention(remainder, self.t, self.v, self.detect, self.target)
                        X_decimal = torch.round((X_decimal - remainder) / cellRange) * mask
                        # Now also consider weight has on/off ratio effects
                        # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                        # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                        remainderQ = (upper - lower) * (remainder - 0) + (
                                    cellRange - 1) * lower  # weight cannot map to 0, but to Gmin
                        remainderQ = remainderQ + remainderQ * torch.normal(0., torch.full(remainderQ.size(), self.vari,
                                                                                           device='cuda'))
                        outputPartial = F.linear(inputB, remainderQ * mask, self.bias)
                        outputDummyPartial = F.linear(inputB, dummyP * mask, self.bias)
                        # Add ADC quanization effects here !!!
                        outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                        outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                        scaler = cellRange ** k
                        outputP = outputP + outputPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                        outputD = outputD + outputDummyPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                    scalerIN = 2 ** z
                    outputIN = outputIN + (outputP - outputD) * scalerIN
                output = output + outputIN / (2 ** bitActivation)
            else:
                inputQ = torch.round((2 ** bitActivation - 1) / 1 * (input - 0) + 0)
                outputIN = torch.zeros_like(outputOrignal)
                for z in range(bitActivation):
                    inputB = torch.fmod(inputQ, 2)
                    inputQ = torch.round((inputQ - inputB) / 2)
                    outputP = torch.zeros_like(outputOrignal)
                    for s in range(numSubArray):
                        mask = torch.zeros_like(weight)
                        mask[:, (s * self.subArray):(s + 1) * self.subArray] = 1
                        # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                        X_decimal = torch.round((2 ** bitWeight - 1) / 2 * (weight + 1) + 0) * mask
                        outputSP = torch.zeros_like(outputOrignal)
                        outputD = torch.zeros_like(outputOrignal)
                        for k in range(int(bitWeight / self.cellBit)):
                            remainder = torch.fmod(X_decimal, cellRange) * mask
                            # retention
                            remainder = wage_quantizer.Retention(remainder, self.t, self.v, self.detect, self.target)
                            X_decimal = torch.round((X_decimal - remainder) / cellRange) * mask
                            # Now also consider weight has on/off ratio effects
                            # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                            # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
                            remainderQ = (upper - lower) * (remainder - 0) + (
                                        cellRange - 1) * lower  # weight cannot map to 0, but to Gmin
                            remainderQ = remainderQ + remainderQ * torch.normal(0.,
                                                                                torch.full(remainderQ.size(), self.vari,
                                                                                           device='cuda'))
                            outputPartial = F.linear(inputB, remainderQ * mask, self.bias)
                            outputDummyPartial = F.linear(inputB, dummyP * mask, self.bias)
                            # Add ADC quanization effects here !!!
                            outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                            outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial,
                                                                                   self.ADCprecision)
                            scaler = cellRange ** k
                            outputSP = outputSP + outputPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                            outputD = outputD + outputDummyPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                        outputSP = outputSP - outputD  # minus dummy column
                        outputP = outputP + outputSP
                    scalerIN = 2 ** z
                    outputIN = outputIN + outputP * scalerIN
                output = output + outputIN / (2 ** bitActivation)
            output = output / (2 ** bitWeight)

        elif self.inference == 1:
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
            weight = wage_quantizer.Retention(weight, self.t, self.v, self.detect, self.target)
            input = wage_quantizer.Q(input, self.wl_input)
            output = F.linear(input, weight, self.bias)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        else:
            # original WAGE QCov2d
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
            weight = wage_quantizer.Retention(weight, self.t, self.v, self.detect, self.target)
            input = wage_quantizer.Q(input, self.wl_input)
            output = F.linear(input, weight, self.bias)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)

            # weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            # weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
            # weight = wage_quantizer.Retention(weight, self.t, self.v, self.detect, self.target)
            # output = F.linear(input, weight, self.bias)

        output = output / self.scale
        output = wage_quantizer.WAGEQuantizer_f(output, self.wl_activate, self.wl_error)

        return output

