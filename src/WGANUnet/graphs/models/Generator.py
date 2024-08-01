import torch.nn as nn
import torch


from src_c.GAN.settings import *


# class Generator(nn.Module):
#     def __init__(self, ngpu):
#         super(Generator, self).__init__()

#         nz = 100
#         ngf = 64
#         nc = 3


#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )

#     def forward(self, input):
#         return self.main(input)



# class Encoder(nn.Module):
#     def __init__(self, ngpu):
#         super(Encoder, self).__init__()

#         ndf = 64
#         nc = 3

#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(ndf * 8, 30, 4, 1, 0, bias=False),
#             # nn.Sigmoid()
#         )

#     def forward(self, input):
#         return self.main(input)


class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()

        ndf = 64
        nc = 3

        self.ngpu = ngpu
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.relu1=nn.LeakyReLU(0.2, inplace=False)
            # state size. (ndf) x 32 x 32
        self.conv2=nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.batch1=nn.BatchNorm2d(ndf * 2)
        self.relu2=nn.LeakyReLU(0.2, inplace=False)
            # state size. (ndf*2) x 16 x 16
        self.conv3=nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.batch2=nn.BatchNorm2d(ndf * 4)
        self.relu3=nn.LeakyReLU(0.2, inplace=False)
            # state size. (ndf*4) x 8 x 8
        self.conv4=nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.batch3=nn.BatchNorm2d(ndf * 8)
        self.relu4=nn.LeakyReLU(0.2, inplace=False)
            # state size. (ndf*8) x 4 x 4
        self.conv5=nn.Conv2d(ndf * 8, NON_RANDOM_PART_SIZE, 4, 1, 0, bias=False)
            # nn.Sigmoid()

    def forward(self, input):
        outs = []
        input=self.conv1(input)
        input=self.relu1(input)
        outs.append(input)
            # state size. (ndf) x 32 x 32
        input=self.conv2(input)
        input=self.batch1(input)
        input=self.relu2(input)
        outs.append(input)
            # state size. (ndf*2) x 16 x 16
        input=self.conv3(input)
        input=self.batch2(input)
        input=self.relu3(input)
        outs.append(input)
            # state size. (ndf*4) x 8 x 8
        input=self.conv4(input)
        input=self.batch3(input)
        input=self.relu4(input)
        outs.append(input)
            # state size. (ndf*8) x 4 x 4
        input=self.conv5(input)

        return input,outs


class Decoder(nn.Module):
    def __init__(self, ngpu):
        super(Decoder, self).__init__()

        nz = LATENT_VECTOR_SIZE
        ngf = 64
        nc = 3


        self.ngpu = ngpu

        self.conv1 = nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False)
        self.batch1=nn.BatchNorm2d(ngf * 8)
        self.relu1=nn.ReLU(False)
            # state size. (ndf) x 32 x 32
        self.conv2=nn.ConvTranspose2d(ngf * 16, ngf * 4, 4, 2, 1, bias=False)
        self.batch2=nn.BatchNorm2d(ngf * 4)
        self.relu2=nn.ReLU(False)
        

        self.conv3=nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1, bias=False)
        self.batch3=nn.BatchNorm2d(ngf * 2)
        self.relu3=nn.ReLU(False)
            # state size. (ndf*4) x 8 x 8
        self.conv4=nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1, bias=False)
        self.batch4=nn.BatchNorm2d(ngf)
        self.relu4=nn.ReLU(False)
            # state size. (ndf*8) x 4 x 4
        self.conv5=nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False)
        self.tan = nn.Tanh()

    def forward(self, input, middle_layers):
        input=self.conv1(input)
        input=self.batch1(input)
        input=self.relu1(input)

        input = torch.cat((input, middle_layers[-1]), dim=1)
    
        input=self.conv2(input)
        input=self.batch2(input)
        input=self.relu2(input)


        input = torch.cat((input, middle_layers[-2]), dim=1)


        input=self.conv3(input)
        input=self.batch3(input)
        input=self.relu3(input)

        input = torch.cat((input, middle_layers[-3]), dim=1)

        input=self.conv4(input)
        input=self.batch4(input)
        input=self.relu4(input)

        input = torch.cat((input, middle_layers[-4]), dim=1)

        input=self.conv5(input)
        input = self.tan(input)

        return input