import argparse
import itertools
import gc

import torch
from torch.utils.data import DataLoader
from torch.autograd import  Variable
import torchvision.transforms as transforms
from PIL import  Image

from models import Generator
from models import Discriminator
from models import GANLoss
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import ImageDataset
from utils import weights_init_normal

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/apple2orange/', help='directory of dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU')
parser.add_argument('--n_cpu', type=int, default=8, help='number of CPU threads to use during batch generation')
opt = parser.parse_args()
print(opt)

##################################
######## Variables ########
#Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
#print(netG_A2B)
netD_A = Discriminator(opt.input_nc)
#print(netD_A)
netD_B = Discriminator(opt.output_nc)


if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

#Optimizer
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(),netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(list(netD_A.parameters()),
                                 lr = opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(list(netD_B.parameters()),
                                 lr = opt.lr, betas=(0.5, 0.999))

#LR schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

#Inputs and targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

#Losses
criterion_GAN = GANLoss(tensor=Tensor)
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

#Dataset loader
transform = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
              transforms.RandomCrop(opt.size),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transform=transform, unaligned=True),
                        batch_size=opt.batchSize,
                        shuffle=True,
                        num_workers=opt.n_cpu)

#Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################

######### Training ############
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        #model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ####### Generators A2B and B2A #######
        optimizer_G.zero_grad()

        #Identity loss
        #G_A2B(B) = B
        same_B = netG_A2B.forward(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        #G_B2A(A) = A
        same_A = netG_B2A.forward(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        #GAN loss
        fake_B = netG_A2B.forward(real_A)
        pred_fake = netD_B.forward(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, True)

        fake_A = netG_B2A.forward(real_B)
        pred_fake = netD_A.forward(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, True)

        #Cycle loss
        recovered_A = netG_B2A.forward(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B.forward(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        #Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_BAB + loss_cycle_ABA
        loss_G.backward()

        #Optimize
        optimizer_G.step()
        ##################################

        ####### Discriminator A and B #######
        ### Discriminator A ###
        optimizer_D_A.zero_grad()

        #Real loss
        pred_real = netD_A.forward(real_A)
        loss_D_real = criterion_GAN(pred_real, True)

        #Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A.forward(fake_A.detach()) #阻断反向传播
        loss_D_fake = criterion_GAN(pred_fake, False)

        #Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        #Optimizer
        optimizer_D_A.step()
        ##########

        ### Discriminator B ###
        optimizer_D_B.zero_grad()

        #Real loss
        pred_real = netD_B.forward(real_B)
        loss_D_real = criterion_GAN(pred_real, True)

        #Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B.forward(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, False)

        #Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        #Optimizer
        optimizer_D_B.step()
        ########
        ##########################

        #Progress report
        logger.log({'loss_G' : loss_G,
                    'loss_G_identity' : (loss_identity_A + loss_identity_B),
                    'loss_G_GAN' : (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle' : (loss_cycle_ABA + loss_cycle_BAB),
                    'loss_D' : (loss_D_A + loss_D_B)},
                    images={'real_A' : real_A,
                            'real_B' : real_B,
                            'fake_A' : fake_A,
                            'fake_B' : fake_B})

    #Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    #Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')
########################################











