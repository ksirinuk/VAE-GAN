import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from KS_lib.general import matlab


class CPPNVAE(torch.nn.Module):
    def __init__(self,
                 batch_size=1, z_dim=32,
                 x_dim=26, y_dim=26, c_dim=1, scale=8.0,
                 learning_rate_g=0.01, learning_rate_d=0.001, learning_rate_vae=0.0001, beta1=0.9, net_size_g=128,
                 net_depth_g=4,
                 net_size_q=512, keep_prob=1.0, df_dim=24, model_name="cppnvae"):
        super(CPPNVAE, self).__init__()
        """
    
        Args:
        z_dim               dimensionality of the latent vector
        x_dim, y_dim        default resolution of generated images for training
        c_dim               1 for monotone, 3 for colour
        learning_rate       learning rate for the generator
                      _d    learning rate for the discriminiator
                      _vae  learning rate for the variational autoencoder
        net_size_g          number of activations per layer for cppn generator function
        net_depth_g         depth of generator
        net_size_q          number of activations per layer for decoder (real image -> z). 2 layers.
        df_dim              discriminiator is a convnet.  higher -> more activations -> smarter.
        keep_prob           dropout probability
    
        when training, use I used dropout on training the decoder, batch norm on discriminator, nothing on cppn
        choose training parameters so that over the long run, decoder and encoder log errors hover around 0.7 each (so they are at the same skill level)
        while the error for vae should slowly move lower over time with D and G balanced.
    
        """

        self.batch_size = batch_size
        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        self.learning_rate_vae = learning_rate_vae
        self.beta1 = beta1
        self.net_size_g = net_size_g
        self.net_size_q = net_size_q
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.scale = scale
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.net_depth_g = net_depth_g
        self.model_name = model_name
        self.keep_prob = keep_prob
        self.df_dim = df_dim
        self.n_points = x_dim * y_dim

        self.bce_with_logits = torch.nn.BCEWithLogitsLoss()

        # discriminator weights
        self.de_conv1 = torch.nn.Conv2d(self.c_dim, self.df_dim, kernel_size=5, stride=2, padding=1)
        self.de_conv2 = torch.nn.Conv2d(self.df_dim, self.df_dim * 2, kernel_size=5, stride=2, padding=1)
        self.de_conv3 = torch.nn.Conv2d(self.df_dim * 2, self.df_dim * 4, kernel_size=5, stride=2, padding=1)
        self.de_linear = torch.nn.Linear(self.df_dim * 4 * 2 * 2, 1)

        self.de_bn1 = torch.nn.BatchNorm2d(self.batch_size)
        self.de_bn2 = torch.nn.BatchNorm2d(self.batch_size)

        # encoder weights
        self.en_linear1 = torch.nn.Linear(self.n_points * self.c_dim, self.net_size_q)
        self.en_linear2 = torch.nn.Linear(self.net_size_q, self.net_size_q)
        self.en_linear_mean = torch.nn.Linear(self.net_size_q, self.z_dim)
        self.en_linear_sigma = torch.nn.Linear(self.net_size_q, self.z_dim)

        # generator weights
        self.ge_fc_x = torch.nn.Linear(1, self.net_size_g)
        self.ge_fc_y = torch.nn.Linear(1, self.net_size_g)
        self.ge_fc_r = torch.nn.Linear(1, self.net_size_g)
        self.ge_fc_z = torch.nn.Linear(self.z_dim, self.net_size_g)

        self.ge_middle = []
        for i in range(self.net_depth_g):
            self.ge_middle.append(torch.nn.Linear(self.net_size_g, self.net_size_g))
        self.ge_middle = torch.nn.Sequential(*self.ge_middle)
        self.ge_fc_out = torch.nn.Linear(self.net_size_g, self.c_dim)

        # weight initialisation
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight, gain=torch.nn.init.calculate_gain('relu'))
                # torch.nn.init.uniform(m.bias.data, a=-1, b=1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal(m.weight, gain=torch.nn.init.calculate_gain('leaky_relu',0.2))
                # torch.nn.init.uniform(m.bias.data, a=-1, b=1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # grouping variables
        self.q_vars = [param for name, param in self.named_parameters() if 'en' in name]
        self.g_vars = [param for name, param in self.named_parameters() if 'ge' in name]
        self.d_vars = [param for name, param in self.named_parameters() if 'de' in name]
        self.vae_vars = self.q_vars + self.g_vars

        # Use ADAM optimizer
        self.d_optimizer = torch.optim.Adam(self.d_vars, lr=self.learning_rate_d)
        self.g_optimizer = torch.optim.Adam(self.g_vars, lr=self.learning_rate_g)
        self.vae_optimizer = torch.optim.Adam(self.vae_vars, self.learning_rate_vae)

    def forward(self, batch):

        # tf Graph batch of image (batch_size, height, width, depth)
        batch_flatten = batch.permute(0, 2, 3, 1).view(self.batch_size, -1)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        z_mean, z_log_sigma_sq = self.encoder(batch_flatten)

        # Draw one sample z from Gaussian distribution
        eps = torch.FloatTensor(self.batch_size, self.z_dim).normal_()
        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        eps = Variable(eps.type(dtype))

        # z = mu + sigma*epsilon
        z = z_mean + torch.sqrt(torch.exp(z_log_sigma_sq)) * eps

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        G = self.generator(z, gen_x_dim=self.x_dim, gen_y_dim=self.y_dim)
        G = G.permute(0, 3, 1, 2)

        D_right = self.discriminator(batch)  # discriminiator on correct examples
        D_wrong = self.discriminator(G)  # feed generated images into D

        return G, z_mean, z_log_sigma_sq, D_right, D_wrong

    def vae_loss_terms(self, batch, batch_reconstruct, z_mean, z_log_sigma_sq):

        batch_flatten = batch.view(self.batch_size, -1)
        batch_reconstruct_flatten = batch_reconstruct.view(self.batch_size, -1)

        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under thebatch_flatten reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        reconstr_loss = \
            -torch.sum(batch_flatten * torch.log(1e-10 + batch_reconstruct_flatten)
                       + (1 - batch_flatten) * torch.log(1e-10 + 1 - batch_reconstruct_flatten), dim=1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        #     between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * torch.sum(1 + z_log_sigma_sq - z_mean ** 2 - torch.exp(z_log_sigma_sq), dim=1)
        vae_loss = torch.mean(reconstr_loss + latent_loss) / (batch_flatten.shape[1])  # average over batch and pixel

        return vae_loss

    def gan_loss_terms(self, D_right, D_wrong):
        # Define loss function and optimiser
        d_loss_real = self.bce_with_logits(torch.ones_like(D_right), D_right)
        d_loss_fake = self.bce_with_logits(torch.zeros_like(D_wrong), D_wrong)
        d_loss = 1.0 * (d_loss_real + d_loss_fake) / 2.0
        g_loss = 1.0 * self.bce_with_logits(torch.ones_like(D_wrong), D_wrong)

        return d_loss, g_loss

    def balanced_loss(self, batch, batch_reconstruct, z_mean, z_log_sigma_sq, D_right, D_wrong):
        vae_loss = self.vae_loss_terms(batch, batch_reconstruct, z_mean, z_log_sigma_sq)
        _, g_loss = self.gan_loss_terms(D_right, D_wrong)

        return 1.0 * g_loss + 1.0 * vae_loss  # can try to weight these.

    def coordinates(self, batch_size, x_dim, y_dim, scale):
        n_points = x_dim * y_dim
        x_range = (np.arange(x_dim) / np.float(x_dim - 1) - 0.5) * 2.0 * scale
        y_range = (np.arange(y_dim) / np.float(y_dim - 1) - 0.5) * 2.0 * scale
        x_mat, y_mat = np.meshgrid(x_range, y_range)
        r_mat = np.sqrt(x_mat ** 2 + y_mat ** 2)
        x_vec = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
        y_vec = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
        r_vec = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
        return x_vec, y_vec, r_vec

    def encoder(self, batch_flatten):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        H1 = F.dropout(F.softplus(self.en_linear1(batch_flatten)), self.keep_prob)
        H2 = F.dropout(F.softplus(self.en_linear2(H1)), self.keep_prob)

        z_mean = self.en_linear_mean(H2)
        z_log_sigma_sq = self.en_linear_sigma(H2)

        return z_mean, z_log_sigma_sq

    def discriminator(self, image):

        h0 = F.leaky_relu(self.de_conv1(image), negative_slope=0.2)
        h1 = F.leaky_relu(self.de_conv2(h0), negative_slope=0.2)
        h2 = F.leaky_relu(self.de_conv3(h1), negative_slope=0.2)
        h3 = self.de_linear(h2.view(self.batch_size, - 1))

        return F.sigmoid(h3)

    def generator(self, z, gen_x_dim, gen_y_dim):
        batch_size = z.shape[0]
        z_dim = z.shape[1]

        gen_n_points = gen_x_dim * gen_y_dim

        x_vec, y_vec, r_vec = self.coordinates(batch_size, gen_x_dim, gen_y_dim, self.scale)

        # inputs to cppn, like coordinates and radius from centre
        x = torch.from_numpy(x_vec)
        y = torch.from_numpy(y_vec)
        r = torch.from_numpy(r_vec)

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # rescale z and reshape z
        z_scaled = z.view((batch_size, 1, z_dim)) * Variable(torch.ones((gen_n_points, 1)).type(dtype)) * self.scale
        z_unroll = z_scaled.view(batch_size * gen_n_points, z_dim)

        # reshape x,y,r
        x_unroll = x.view(batch_size * gen_n_points, 1)
        y_unroll = y.view(batch_size * gen_n_points, 1)
        r_unroll = r.view(batch_size * gen_n_points, 1)

        # network
        U = self.ge_fc_z(z_unroll) + \
            self.ge_fc_x(Variable(x_unroll.type(dtype), requires_grad=False)) + \
            self.ge_fc_y(Variable(y_unroll.type(dtype), requires_grad=False)) + \
            self.ge_fc_r(Variable(r_unroll.type(dtype), requires_grad=False))

        H = F.softplus(U)
        for i in range(self.net_depth_g):
            H = torch.tanh(self.ge_middle[i](H))
        output = torch.sigmoid(self.ge_fc_out(H))

        output = output.view(batch_size, gen_y_dim, gen_x_dim, self.c_dim)
        return output

    def encode(self, batch):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        if batch.ndim == 4:
            batch_size = batch.shape[0]
        else:
            batch_size = 1

        batch_flatten = batch.view(batch_size, -1)
        z_mean, _ = self.encoder(batch_flatten)

        return z_mean

    def generate(self, z=None, gen_x_dim=26, gen_y_dim=26):
        """ Generate data by sampling from latent space.

        If z is not None, data for this point in latent space is
        generated. Otherwise, z is drawn from prior in latent
        space.
        """
        if z is None:
            z = np.random.normal(size=(self.batch_size, self.z_dim)).astype(np.float32)
        image = self.generator(torch.from_numpy(z), gen_x_dim, gen_y_dim)
        image = image.cpu().data.numpy()

        return image

