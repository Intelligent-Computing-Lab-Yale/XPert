from utee import misc
print = misc.logger.info
import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import torch

class VGG(nn.Module):
    def __init__(self, args, features, num_classes,logger):
        super(VGG, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.classifier = make_layers([('L', 512, num_classes)],
                                      args, logger)[0]
        # self.classifier = make_layers([('L', 2048, 1024),
        #                                ('L', 1024, num_classes)],
        #                                args, logger)

        # print(self.features)
        # print(self.classifier)

    def forward(self, x, dac_prec, adc_prec):
        count = 0
        # print(highest_adc_param)
        for i in self.features:
            if isinstance(i, FConv2d):
                x = i(x, dac_prec[count], adc_prec[count])  # .detach()
                count += 1
                # print('this')
            else:
                x = i(x)  # .detach()
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, args, logger ):
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
                                 logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                                 wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                 subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                 name = 'Conv'+str(i)+'_', model = args.model)
            elif args.mode == "FP":
                conv2d = FConv2d(in_channels, out_channels, kernel_size=v[2], padding=padding,
                                 logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
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
                             logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
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

    'Custom_Model_3' : [('C', 256, 3, 'same', 8), ('M', 2, 2), ('C', 256, 3, 'same', 8), ('C', 32, 3, 'same', 8), ('C', 16, 3, 'same', 8),
                        ('M', 2, 2), ('C', 32, 3, 'same', 8), ('C', 512, 3, 'same', 8), ('C', 256, 3, 'same', 8), ('M', 2, 2), ('C', 256, 3, 'same', 8),
                        ('C', 16, 3, 'same', 8), ('C', 512, 3, 'same', 8), ('C', 64, 3, 'same', 8), ('C', 32, 3, 'same', 8)],

    '14.9mm2_model': [('C', 16, 3, 'same', 8), ('M', 2, 2), ('C', 16, 3, 'same', 8), ('C', 32, 3, 'same', 8),
                      ('C', 128, 3, 'same', 8), ('M', 2, 2), ('C', 256, 3, 'same', 8), ('C', 128, 3, 'same', 8),
                      ('C', 128, 3, 'same', 8), ('M', 2, 2), ('C', 256, 3, 'same', 8), ('C', 16, 3, 'same', 8),
                      ('C', 16, 3, 'same', 8), ('C', 256, 3, 'same', 8), ('C', 32, 3, 'same', 8)],


    '80mm2_model': [('C', 256, 3, 'same', 8), ('C', 512, 3, 'same', 8), ('M', 2, 2), ('C', 512, 3, 'same', 8), ('C', 128, 3, 'same', 8), ('M', 2, 2), ('C', 256, 3, 'same', 8), ('C', 256, 3, 'same', 8), ('C', 256, 3, 'same', 8), ('M', 2, 2), ('C', 256, 3, 'same', 8), ('C', 64, 3, 'same', 8), ('C', 256, 3, 'same', 8), ('M', 2, 2), ('C', 128, 3, 'same', 8), ('C', 128, 3, 'same', 8), ('C', 512, 3, 'same', 8)],
    'XPertNet_50': [('C', 64, 3, 'same', 8), ('C', 128, 3, 'same', 8), ('M', 2, 2), ('C', 64, 3, 'same', 8), ('C', 512, 3, 'same', 8), ('M', 2, 2), ('C', 512, 3, 'same', 8), ('C', 128, 3, 'same', 8), ('C', 128, 3, 'same', 8), ('M', 2, 2), ('C', 64, 3, 'same', 8), ('C', 512, 3, 'same', 8), ('C', 128, 3, 'same', 8), ('M', 2, 2), ('C', 64, 3, 'same', 8), ('C', 64, 3, 'same', 8), ('C', 64, 3, 'same', 8)]


}

def vgg8( args, logger, pretrained=None):
    cfg = cfg_list['vgg16']
    layers = make_layers(cfg, args, logger)
    model = VGG(args,layers, num_classes=10,logger = logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


