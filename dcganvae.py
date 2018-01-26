import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn


# custom weights initialization called on netG and netD
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


class _Sampler(torch.nn.Module):
    def __init__(self):
        super(_Sampler, self).__init__()

    def forward(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # calculate the STDEV
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # random normalized noise
        else:
            eps = torch.FloatTensor(std.size()).normal_()  # random normalized noise
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


class _Encoder(torch.nn.Module):
    def __init__(self, imageSize, nz, ngf, nc):
        super(_Encoder, self).__init__()

        n = np.sqrt(imageSize)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        self.conv1 = nn.Conv2d(ngf * 2 ** (n - 5), nz, 4)
        self.conv2 = nn.Conv2d(ngf * 2 ** (n - 5), nz, 4)

        self.encoder = nn.Sequential()
        # input is (nc) x 64 x 64
        self.encoder.add_module('input-conv', nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        self.encoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))
        for i in range(n - 5):
            # state size. (ngf) x 32 x 32
            self.encoder.add_module('pyramid.{0}-{1}.conv'.format(ngf * 2 ** i, ngf * 2 ** (i + 1)),
                                    nn.Conv2d(ngf * 2 ** (i), ngf * 2 ** (i + 1), 4, 2, 1, bias=False))
            self.encoder.add_module('pyramid.{0}.batchnorm'.format(ngf * 2 ** (i + 1)),
                                    nn.BatchNorm2d(ngf * 2 ** (i + 1)))
            self.encoder.add_module('pyramid.{0}.relu'.format(ngf * 2 ** (i + 1)), nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf*8) x 4 x 4
        self.apply(weights_init)

    def forward(self, input):
        output = self.encoder(input)
        return self.conv1(output), self.conv2(output)


class _netG(nn.Module):
    def __init__(self, imageSize, ngpu, nz, ngf, nc, lr, beta1):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.encoder = _Encoder(imageSize, nz, ngf, nc)
        self.sampler = _Sampler()

        n = np.sqrt(imageSize)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        self.decoder = nn.Sequential()
        # input is Z, going into a convolution
        self.decoder.add_module('input-conv', nn.ConvTranspose2d(nz, ngf * 2 ** (n - 5), 4, 1, 0, bias=False))
        self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(ngf * 2 ** (n - 5)))
        self.decoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf * 2**(n-3)) x 4 x 4

        for i in range(n - 5, 0, -1):
            self.decoder.add_module('pyramid.{0}-{1}.conv'.format(ngf * 2 ** i, ngf * 2 ** (i - 1)),
                                    nn.ConvTranspose2d(ngf * 2 ** i, ngf * 2 ** (i - 1), 4, 2, 1, bias=False))
            self.decoder.add_module('pyramid.{0}.batchnorm'.format(ngf * 2 ** (i - 1)),
                                    nn.BatchNorm2d(ngf * 2 ** (i - 1)))
            self.decoder.add_module('pyramid.{0}.relu'.format(ngf * 2 ** (i - 1)), nn.LeakyReLU(0.2, inplace=True))

        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        self.decoder.add_module('output-tanh', nn.Tanh())
        self.apply(weights_init)

        self.MSECriterion = nn.MSELoss()
        self.BCECriterion = nn.BCELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            output = nn.parallel.data_parallel(self.sampler, output, range(self.ngpu))
            output = nn.parallel.data_parallel(self.decoder, output, range(self.ngpu))
        else:
            mu, logvar = self.encoder(input)
            output = self.sampler(mu, logvar)
            output = self.decoder(output)
        return output

    def VAELoss(self, recon, input, mu, logvar):
        MSEerr = self.MSECriterion(recon, input)

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return KLD + MSEerr

    def GLoss(self, input):
        return self.BCECriterion(input, torch.ones_like(input))


class _netD(nn.Module):
    def __init__(self, imageSize, ngpu, ngf, ndf, nc, lr, beta1):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        n = np.sqrt(imageSize)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)
        self.main = nn.Sequential()

        # input is (nc) x 64 x 64
        self.main.add_module('input-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.main.add_module('relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ndf) x 32 x 32
        for i in range(n - 5):
            self.main.add_module('pyramid.{0}-{1}.conv'.format(ngf * 2 ** (i), ngf * 2 ** (i + 1)),
                                 nn.Conv2d(ndf * 2 ** (i), ndf * 2 ** (i + 1), 4, 2, 1, bias=False))
            self.main.add_module('pyramid.{0}.batchnorm'.format(ngf * 2 ** (i + 1)), nn.BatchNorm2d(ndf * 2 ** (i + 1)))
            self.main.add_module('pyramid.{0}.relu'.format(ngf * 2 ** (i + 1)), nn.LeakyReLU(0.2, inplace=True))

        self.main.add_module('output-conv', nn.Conv2d(ndf * 2 ** (n - 5), 1, 4, 1, 0, bias=False))
        self.main.add_module('output-sigmoid', nn.Sigmoid())

        self.apply(weights_init)

        self.BCEcriterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

    def DLoss(self, real, fake):
        errD_real = self.BCEcriterion(real, torch.ones_like(real))
        errD_fake = self.BCEcriterion(fake, torch.zeros_like(fake))
        return errD_real + errD_fake
