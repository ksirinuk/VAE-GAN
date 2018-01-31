from __future__ import print_function
import argparse
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from dcganvae import _netG, _netD
import vae_dataset as vae_data
import torchvision
import numpy as np
from KS_lib.general import matlab
from KS_lib import KSimage
import time
from tqdm import tqdm
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', default=4000, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--patch_size', default=64, type=int)
parser.add_argument('--dir', default='Photos', type=str)
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')

parser.add_argument('--learning_rate_g', default=0.0002, type=float)
parser.add_argument('--learning_rate_d', default=0.0002, type=float)
parser.add_argument('--beta1', default=0.5, type=float)

parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--workers', default=6, type=int)
parser.add_argument('--checkpoint_folder', default='checkpoints', type=str)
parser.add_argument('--resume', default='checkpoints/checkpoint_436.pth', type=str)

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)

parser.add_argument('--train_vae', default=False, type=bool)


def create_dataset(args, mode):
    train_transform = torchvision.transforms.Compose([
        vae_data.Scale(opt.patch_size),
        vae_data.CenterCrop(opt.patch_size),
        vae_data.RandomHorizontalFlip(),
        vae_data.RandomVerticalFlip(),
        vae_data.RandomTransposeFlip(),
        vae_data.Convert(),
    ])

    return vae_data.VAEDataset(args.dir, mode, train_transform)


def main():
    global opt
    opt = parser.parse_args()

    # initialize CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    if torch.cuda.is_available():
        torch.randn(8).cuda()

    # create data loader
    train_dataset = create_dataset(opt, 'train')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.workers)

    # def show_landmarks_batch(sample_batched):
    #     """Show image with landmarks for a batch of samples."""
    #     images_batch = sample_batched
    #
    #     grid = torchvision.utils.make_grid(images_batch)
    #     x = grid.numpy()
    #     x = x.transpose(1, 2, 0)
    #     KSimage.imshow(x)
    #
    # # for dat in train_loader:
    # show_landmarks_batch(next(iter(train_loader)))

    # create model
    netG = _netG(opt.patch_size, opt.ngpu, opt.nz, opt.ngf, opt.nc, opt.learning_rate_g, opt.beta1)
    netD = _netD(opt.patch_size, opt.ngpu, opt.ngf, opt.ndf, opt.nc, opt.learning_rate_d, opt.beta1)

    if torch.cuda.is_available():
        netD.cuda()
        netG.cuda()

    # create a checkpoint folder if not exists
    if not os.path.exists(opt.checkpoint_folder):
        os.makedirs(opt.checkpoint_folder)

    # optionally resume from a checkpoint
    start_epoch = 0
    train_vae_loss = []
    train_rep_loss = []
    train_d_loss = []
    train_g_loss = []
    train_D_x = []
    train_D_G_z1 = []
    train_D_G_z2 = []
    train_stddev = []
    train_update_D = []
    train_update_G = []

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG.load_state_dict(checkpoint['netG_state_dict'])
            netD.load_state_dict(checkpoint['netD_state_dict'])
            start_epoch = checkpoint['epoch']
            train_vae_loss = checkpoint['train_vae_loss']
            train_rep_loss = checkpoint['train_rep_loss']
            train_d_loss = checkpoint['train_d_loss']
            train_g_loss = checkpoint['train_g_loss']
            train_D_x = checkpoint['train_D_x']
            train_D_G_z1 = checkpoint['train_D_G_z1']
            train_D_G_z2 = checkpoint['train_D_G_z2']
            train_stddev = checkpoint['train_stddev']
            train_update_D = checkpoint['train_update_D']
            train_update_G = checkpoint['train_update_G']

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    cudnn.benchmark = True

    for epoch in range(start_epoch, opt.epochs):
        # train for one epoch
        # stddev = 0.1 * (1 - int(epoch/100)) if epoch <= 100 else 0
        stddev = 0.0

        vae_loss, rep_loss, d_loss, g_loss, D_x, D_G_z1, D_G_z2, \
        t_time, update_D, update_G = train(train_loader, netG, netD, stddev)

        train_vae_loss.append(vae_loss)
        train_rep_loss.append(rep_loss)
        train_d_loss.append(d_loss)
        train_g_loss.append(g_loss)
        train_D_x.append(D_x)
        train_D_G_z1.append(D_G_z1)
        train_D_G_z2.append(D_G_z2)
        train_stddev.append(stddev)
        train_update_D.append(update_D)
        train_update_G.append(update_G)

        print_str = 'epoch: %d ' \
                    'train_vae_loss: %.3f ' \
                    'train_rep_loss: %.3f ' \
                    'train_d_loss: %.3f ' \
                    'train_g_loss: %.3f ' \
                    'train_D_x: %.3f ' \
                    'train_D_G_z1: %.3f ' \
                    'train_D_G_z2: %.3f ' \
                    'train_time: %.3f ' \
                    'train_update_D: %d ' \
                    'train_update_G: %d '

        print(print_str % (epoch, vae_loss, rep_loss, d_loss, g_loss, D_x, D_G_z1, D_G_z2, t_time,
                           update_D, update_G))

        # save checkpoint
        save_name = os.path.join(opt.checkpoint_folder,
                                 'checkpoint_' + str(epoch) + '.pth')
        save_variable_name = os.path.join(opt.checkpoint_folder,
                                          'variables.mat')

        save_checkpoint({
            'epoch': epoch + 1,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'train_vae_loss': train_vae_loss,
            'train_rep_loss': train_rep_loss,
            'train_d_loss': train_d_loss,
            'train_g_loss': train_g_loss,
            'train_D_x': train_D_x,
            'train_D_G_z1': train_D_G_z1,
            'train_D_G_z2': train_D_G_z2,
            'train_stddev': train_stddev,
            'train_update_D': train_update_D,
            'train_update_G': train_update_G},
            {'train_vae_loss': np.array(train_vae_loss),
             'train_rep_loss': np.array(train_rep_loss),
             'train_d_loss': np.array(train_d_loss),
             'train_g_loss': np.array(train_g_loss),
             'train_D_x': np.array(train_D_x),
             'train_D_G_z1': np.array(train_D_G_z1),
             'train_D_G_z2': np.array(train_D_G_z2),
             'train_stddev': np.array(train_stddev),
             'train_update_D': np.array(train_update_D),
             'train_update_G': np.array(train_update_G)},
            filename=save_name,
            variable_name=save_variable_name)


def train(train_loader, netG, netD, stddev):
    vae_loss_meter = AverageMeter()
    rep_loss_meter = AverageMeter()
    g_loss_meter = AverageMeter()
    d_loss_meter = AverageMeter()
    time_meter = AverageMeter()
    D_x = AverageMeter()
    D_G_z1 = AverageMeter()
    D_G_z2 = AverageMeter()

    vae_loss_meter.reset()
    rep_loss_meter.reset()
    g_loss_meter.reset()
    d_loss_meter.reset()
    time_meter.reset()
    D_x.reset()
    D_G_z1.reset()
    D_G_z2.reset()

    # switch to train mode
    netG.train()
    netD.train()

    t = time.time()

    ############################
    # (3) Update G network: maximize log(D(G(z)))
    ###########################
    input = next(iter(train_loader))
    input = input.cuda(async=True)
    input_var = Variable(input, requires_grad=False)
    recon = netG(input_var)
    fake, _ = netD(recon)
    g_loss = netG.GLoss(fake)

    update_D = 0
    update_G = 0
    for count, input in enumerate(train_loader):
        sub_time = time.time()
        input = input.cuda(async=True)
        input_var = Variable(input, requires_grad=False)

        if not opt.train_vae:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            if torch.cuda.is_available():
                real, _ = netD(input_var + Variable(torch.randn(input_var.size()).cuda() * stddev))
            else:
                real, _ = netD(input_var + Variable(torch.randn(input_var.size()) * stddev))

            # train on fake
            if torch.cuda.is_available():
                noise = Variable(torch.cuda.FloatTensor(input_var.size(0), opt.nz, 1, 1).normal_(0, 1))
            else:
                noise = Variable(torch.FloatTensor(input_var.size(0), opt.nz, 1, 1).normal_(0, 1))
            gen = netG.decoder(noise)
            fake, _ = netD(gen)
            d_loss, d_loss_real, d_loss_fake = netD.DLoss(real, fake)

            if g_loss.data[0] < 0.7 or d_loss_real.data[0] > 1.0 or d_loss_fake.data[0] > 1.0:
            # if d_loss_real.data[0] > g_loss.data[0] or d_loss_fake.data[0] > g_loss.data[0]: # bad idea because  g will not frequently update
                netD.optimizer.zero_grad()
                d_loss.backward()
                netD.optimizer.step()
                update_D += 1

            # measure accuracy and record loss
            d_loss_meter.update(d_loss.data[0], input.size(0))
            D_x.update(real.data.mean(), real.size(0))
            D_G_z1.update(fake.data.mean(), fake.size(0))

        ############################
        # (2) Update G network: VAE
        ############################
        mu, logvar = netG.encoder(input_var)
        z = netG.sampler(mu, logvar)
        recon = netG.decoder(z)
        vae_loss = netG.VAELoss(recon, input_var, mu, logvar)
        if not opt.train_vae:
            _, feat_response_recon = netD(recon)
            _, feat_response_input = netD(input_var)
            rep_loss = netG.RepLoss(feat_response_recon, feat_response_input)
            visual_loss = vae_loss + rep_loss
        else:
            visual_loss = vae_loss

        netG.optimizer.zero_grad()
        visual_loss.backward()
        netG.optimizer.step()

        # measure accuracy and record loss
        vae_loss_meter.update(vae_loss.data[0], input.size(0))
        rep_loss_meter.update(rep_loss.data[0], input.size(0))

        if not opt.train_vae:
            ############################
            # (3) Update G network: maximize log(D(G(z)))
            ###########################
            recon = netG(input_var)
            fake, _ = netD(recon)
            g_loss = netG.GLoss(fake)

            if d_loss_real.data[0] < 0.7 or d_loss_fake.data[0] < 0.7 or g_loss.data[0] > 1.0:
                netG.optimizer.zero_grad()
                g_loss.backward()
                netG.optimizer.step()
                update_G += 1

            # measure accuracy and record loss
            g_loss_meter.update(g_loss.data[0], input.size(0))
            D_G_z2.update(fake.data.mean(), fake.size(0))

        if count%50 == 0:
            grid = torchvision.utils.make_grid(input_var.cpu().data)
            x = grid.numpy()
            x = x.transpose(1, 2, 0)
            x = x * 255
            x = x.clip(0, 255)
            x = x.astype(np.uint8)
            KSimage.imwrite(x, 'input.png')

            grid = torchvision.utils.make_grid(recon.cpu().data)
            x = grid.numpy()
            x = x.transpose(1, 2, 0)
            x = x * 255
            x = x.clip(0, 255)
            x = x.astype(np.uint8)
            KSimage.imwrite(x, 'gen.png')

        if not opt.train_vae:
            print_str = 'count: %d ' \
                        'batch_vae_loss: %.3f ' \
                        'batch_rep_loss: %.3f ' \
                        'batch_d_loss: %.3f ' \
                        'batch_g_loss: %.3f ' \
                        'avg_D_x: %.3f ' \
                        'avg_D_G_z1: %.3f ' \
                        'avg_D_G_z2: %.3f ' \
                        'time: %.3f ' \
                        'update_D: %d ' \
                        'update_G: %d '

            print(print_str % (count, vae_loss.data[0], rep_loss.data[0], d_loss.data[0], g_loss.data[0],
                               D_x.avg, D_G_z1.avg, D_G_z2.avg, time.time() - sub_time, update_D, update_G))
        else:
            print_str = 'count: %d ' \
                       'batch_vae_loss: %.3f ' \
                       'time: %.3f '

            print(print_str % (count, vae_loss.data[0], time.time() - sub_time))

    # measure elapsed time
    time_meter.update(time.time() - t)

    return vae_loss_meter.avg, rep_loss_meter.avg, d_loss_meter.avg, g_loss_meter.avg, \
           D_x.avg, D_G_z1.avg, D_G_z2.avg, time_meter.sum, update_D, update_G


def save_checkpoint(state,
                    variables,
                    filename='checkpoint.pth.tar', variable_name='variables.mat'):
    torch.save(state, filename)
    matlab.save(variable_name, variables)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
