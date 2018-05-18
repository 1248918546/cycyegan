import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import functools

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(GlobalGenerator, self).__init__()

        #Input convolution layer
        model = [  nn.ReflectionPad2d(3),
                   nn.Conv2d(input_nc, 64, 7),
                   nn.InstanceNorm2d(64),
                   nn.ReLU(inplace=True)  ]

        #Encoding
        #Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(3):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)  ]
            in_features = out_features
            out_features = in_features * 2

        #Residual block
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        #Decoding
        #Upsampling
        out_features = in_features // 2
        for _ in range(3):
            model += [ nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(out_features),
                       nn.ReLU(inplace=True)  ]
            in_features = out_features
            out_features = in_features // 2

        #Output convolution layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh()  ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

    def __len__(self):
        return len(self._modules)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        model_global = GlobalGenerator(input_nc, output_nc, n_residual_blocks).model
        model_global = [model_global[i] for i in range(len(model_global) - 3)] #get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ####Local enhancer layers####
        #Downsampling
        in_features_global = 32
        out_features_global = in_features_global * 2
        model_downsample = [ nn.ReflectionPad2d(3),
                             nn.Conv2d(input_nc, in_features_global, 7),
                             nn.InstanceNorm2d(in_features_global),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_features_global, out_features_global, 3, stride=2, padding=1),
                             nn.InstanceNorm2d(out_features_global),
                             nn.ReLU(inplace=True) ]
        ##Residual block
        model_upsample = []
        for _ in range(n_residual_blocks):
            model_upsample += [ResidualBlock(out_features_global)]

        #Upsampling
        model_upsample += [ nn.ConvTranspose2d(out_features_global, in_features_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(in_features_global),
                           nn.ReLU(inplace=True) ]
        #Output convolution layer
        model_upsample += [ nn.ReflectionPad2d(3),
                            nn.Conv2d(in_features_global, output_nc, 7),
                            nn.Tanh() ]
        setattr(self, 'model1_1', nn.Sequential(*model_downsample))
        setattr(self, 'model1_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        #Create input pyramid
        input_downsampled = [input]
        input_downsampled.append(self.downsample(input_downsampled[-1]))

        #Output at coarest level
        output_prev = self.model(input_downsampled[-1])

        #Build up one layer at a time
        model_downsample = getattr(self, 'model1_1')
        model_upsample = getattr(self, 'model1_2')
        input_i = input_downsampled[0]
        output_prev = model_upsample(model_downsample(input_i) + output_prev)

        return output_prev

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super(NLayerDiscriminator, self).__init__()

        #Convolution layers
        sequence = [ nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
                      nn.LeakyReLU(0.2, inplace=True) ]
        sequence += [ nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                       nn.InstanceNorm2d(128),
                       nn.LeakyReLU(0.2, inplace=True) ]
        sequence += [ nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                       nn.InstanceNorm2d(256),
                       nn.LeakyReLU(0.2, inplace=True) ]
        sequence += [nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                      nn.InstanceNorm2d(512),
                      nn.LeakyReLU(0.2, inplace=True)]
        #FCN layer
        sequence += [ nn.Conv2d(512, 1, kernel_size=4, padding=1) ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        for i in range(3):
            netD = NLayerDiscriminator(input_nc)
            setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input):
        result = []
        input_downsampled = input
        for i in range(3):
            model = getattr(self, 'layer'+str(3-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != 2:
                input_downsampled = self.downsample(input_downsampled)

        return result

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real = target_real_label
        self.fake = target_fake_label
        self.real_var = None
        self.fake_var = None
        self.Tensor = tensor
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            real_tensor = self.Tensor(input.size()).fill_(1.0)
            self.real_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_var
        else:
            fake_tensor = self.Tensor(input.size()).fill_(0.0)
            self.fake_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_var
        return target_tensor

    def __call__(self, input, target_is_real):
        loss = 0
        for input_i in input:
            pred = input_i[-1]
            target_tensor = self.get_target_tensor(pred, target_is_real)
            loss += self.loss(pred, target_tensor)
        return loss






