from cppnvae import CPPNVAE

import torch
import torch.backends.cudnn as cudnn
import os
import torchvision
import argparse
import time
import numpy as np

from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
from KS_lib.general import matlab
from torchvision.utils import save_image
from KS_lib import KSimage
import vae_dataset as vae_data
from cppnvae import CPPNVAE

import cv2

'''
cppn vae:

compositional pattern-producing generative adversarial network

LOADS of help was taken from:

https://github.com/carpedm20/DCGAN-tensorflow
https://jmetzen.github.io/2015-11-27/vae.html

'''

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--patch_size', default=26, type=int)
parser.add_argument('--dir', default='Photos', type=str)

parser.add_argument('--learning_rate_g', default=0.005, type=float)
parser.add_argument('--learning_rate_d', default=0.001, type=float)
parser.add_argument('--learning_rate_vae', default=0.001, type=float)
parser.add_argument('--keep_prob', default=1.0, type=float)
parser.add_argument('--beta1', default=0.65, type=float)

parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--checkpoint_folder', default='checkpoints', type=str)
parser.add_argument('--resume', default='checkpoints/checkpoint_137.pth', type=str)


def create_dataset(args, mode):
    train_transform = torchvision.transforms.Compose([
        vae_data.RandomScale(),
        vae_data.RandomCrop(args.patch_size),
        # vae_data.RandomHorizontalFlip(),
        # vae_data.RandomVerticalFlip(),
        # vae_data.RandomTransposeFlip(),
        vae_data.Convert(),
    ])

    return vae_data.VAEDataset(args.dir, mode, train_transform)


def main():
    global args
    args = parser.parse_args()

    # initialize CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if torch.cuda.is_available():
        torch.randn(8).cuda()

    # create data loader
    train_dataset = create_dataset(args, 'train')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers)

    # def show_landmarks_batch(sample_batched):
    #     """Show image with landmarks for a batch of samples."""
    #     images_batch = sample_batched
    #
    #     grid = torchvision.utils.make_grid(images_batch)
    #     x = grid.numpy()
    #     x = x.transpose(1,2,0)
    #     # x = np.uint8(x)
    #     plt.imshow(x)
    #
    #     # grid = torchvision.utils.make_grid(landmarks_batch)
    #     # x = grid.numpy()
    #     # x = x.transpose(1, 2, 0)
    #     # x = np.uint8(x)
    #     # plt.imshow(x)
    #
    # # for dat in train_loader:
    # show_landmarks_batch(next(iter(train_loader)))


    # create model
    model = CPPNVAE(batch_size=args.batch_size,
                    learning_rate_g=args.learning_rate_g,
                    learning_rate_d=args.learning_rate_d,
                    learning_rate_vae=args.learning_rate_vae,
                    beta1=args.beta1, keep_prob=args.keep_prob)
    if torch.cuda.is_available():
        model.cuda()

    # create a checkpoint folder if not exists
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)

    # optionally resume from a checkpoint
    start_epoch = 0
    train_vae_loss = []
    train_d_loss = []
    train_g_loss = []
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            train_vae_loss = checkpoint['train_vae_loss']
            train_d_loss = checkpoint['train_d_loss']
            train_g_loss = checkpoint['train_g_loss']
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        vae_loss, d_loss, g_loss, t_time = train(train_loader, model)
        train_vae_loss.append(vae_loss)
        train_d_loss.append(d_loss)
        train_g_loss.append(g_loss)

        print_str = 'epoch: %d ' \
                    'train_vae_loss: %.3f ' \
                    'train_d_loss: %.3f ' \
                    'train_g_loss: %.3f ' \
                    'train_time: %.3f'

        print(print_str % (epoch, vae_loss, d_loss, g_loss, t_time))

        # save checkpoint
        save_name = os.path.join(args.checkpoint_folder,
                                 'checkpoint_' + str(epoch) + '.pth')
        save_variable_name = os.path.join(args.checkpoint_folder,
                                          'variables.mat')

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'train_vae_loss': train_vae_loss,
            'train_d_loss': train_d_loss,
            'train_g_loss': train_g_loss},
            {'train_vae_loss': np.array(train_vae_loss),
             'train_d_loss': np.array(train_d_loss),
             'train_g_loss': np.array(train_g_loss)},
            filename=save_name,
            variable_name=save_variable_name)


def train(train_loader, model):
    vae_loss_meter = AverageMeter()
    g_loss_meter = AverageMeter()
    d_loss_meter = AverageMeter()
    time_meter = AverageMeter()

    vae_loss_meter.reset()
    g_loss_meter.reset()
    d_loss_meter.reset()
    time_meter.reset()

    # switch to train mode
    model.train()

    t = time.time()

    # optimize VAE
    for i in range(100):
        input = next(iter(train_loader))

        input = input.cuda(async=True)
        input_var = Variable(input, requires_grad=False)

        # compute output
        G, z_mean, z_log_sigma_sq, _, _ = model(input_var)
        vae_loss = model.vae_loss_terms(input_var, G, z_mean, z_log_sigma_sq)

        # compute gradient and do SGD step
        model.vae_optimizer.zero_grad()
        vae_loss.backward()
        model.vae_optimizer.step()

        # measure accuracy and record loss
        vae_loss_meter.update(vae_loss.data[0], input.size(0))

    # optimize generator
    for i in range(100):
        input = next(iter(train_loader))

        input = input.cuda(async=True)
        input_var = Variable(input, requires_grad=False)

        # compute output
        G, z_mean, z_log_sigma_sq, D_right, D_wrong = model(input_var)
        balanced_loss = model.balanced_loss(input_var, G, z_mean, z_log_sigma_sq, D_right, D_wrong)

        # compute gradient and do SGD step
        model.g_optimizer.zero_grad()
        balanced_loss.backward()
        model.g_optimizer.step()

        # measure accuracy and record loss
        g_loss_meter.update(balanced_loss.data[0], input.size(0))

        if balanced_loss.data[0] < 0.6:
            break

    grid = torchvision.utils.make_grid(G.cpu().data)
    x = grid.numpy()
    x = x.transpose(1, 2, 0)
    x = x*255
    x = x.astype(np.uint8)
    KSimage.imwrite(x, 'gen.png')

    # optimize discriminator
    input = next(iter(train_loader))

    input = input.cuda(async=True)
    input_var = Variable(input, requires_grad=False)

    # compute output
    _, _, _, D_right, D_wrong = model(input_var)
    d_loss, g_loss = model.gan_loss_terms(D_right, D_wrong)
    d_loss_meter.update(d_loss.data[0], input.size(0))

    # optimize d_loss
    if d_loss.data[0] > 0.6 and g_loss.data[0] < 0.75:
        for i in range(4):
            input = next(iter(train_loader))

            input = input.cuda(async=True)
            input_var = Variable(input, requires_grad=False)

            # compute output
            _, _, _, D_right, D_wrong = model(input_var)
            d_loss, _ = model.gan_loss_terms(D_right, D_wrong)

            # compute gradient and do SGD step
            model.d_optimizer.zero_grad()
            d_loss.backward()
            model.d_optimizer.step()

            # measure accuracy and record loss
            d_loss_meter.update(d_loss.data[0], input.size(0))

            if d_loss.data[0] < 0.6:
                break

    # measure elapsed time
    time_meter.update(time.time() - t)

    return vae_loss_meter.avg, d_loss_meter.avg, g_loss_meter.avg, time_meter.sum


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
