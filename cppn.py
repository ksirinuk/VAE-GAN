import numpy as np
import torch
# from KS_lib import KSimage
from torch.autograd import Variable


class CPPN(torch.nn.Module):
    def __init__(self, batch_size=1, z_dim=32, x_dim=256, y_dim=256, c_dim=1, scale=8.0, net_size=32):
        super(CPPN, self).__init__()
        self.batch_size = batch_size
        self.z_dim = int(z_dim)
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.c_dim = int(c_dim)
        self.scale = float(scale)
        self.net_size = int(net_size)
        self.n_points = self.x_dim * self.y_dim

        x_vec, y_vec, r_vec = self._coordinates()
        self.x_vec = torch.from_numpy(x_vec)
        self.y_vec = torch.from_numpy(y_vec)
        self.r_vec = torch.from_numpy(r_vec)

        self.fc_x = torch.nn.Linear(1, self.net_size)
        self.fc_y = torch.nn.Linear(1, self.net_size)
        self.fc_r = torch.nn.Linear(1, self.net_size)
        self.fc_z = torch.nn.Linear(self.z_dim, self.net_size)

        self.middle = []
        for i in range(3):
            self.middle.append(torch.nn.Linear(self.net_size, self.net_size))
        self.middle = torch.nn.Sequential(*self.middle)

        self.fc_out = torch.nn.Linear(self.net_size, self.c_dim)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal(m.weight)
                torch.nn.init.uniform(m.bias.data, a=-1, b=1)

    def _coordinates(self):
        x_range = (np.arange(self.x_dim) / np.float(self.x_dim - 1) - 0.5) * 2.0 * self.scale
        y_range = (np.arange(self.y_dim) / np.float(self.y_dim - 1) - 0.5) * 2.0 * self.scale
        x_mat, y_mat = np.meshgrid(x_range, y_range)
        r_mat = np.sqrt(x_mat ** 2 + y_mat ** 2)
        x_vec = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, self.n_points, 1)
        y_vec = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, self.n_points, 1)
        r_vec = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, self.n_points, 1)
        return x_vec, y_vec, r_vec

    def generator(self, z):
        # reshape x,y,r
        x_unroll = self.x_vec.view(self.batch_size * self.n_points, 1)
        y_unroll = self.y_vec.view(self.batch_size * self.n_points, 1)
        r_unroll = self.r_vec.view(self.batch_size * self.n_points, 1)

        # rescale z and reshape z
        z_scaled = z.view((self.batch_size, 1, self.z_dim)) * torch.ones((self.n_points, 1)) * self.scale
        z_unroll = z_scaled.view(self.batch_size * self.n_points, self.z_dim)

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # network
        U = self.fc_z(Variable(z_unroll.type(dtype), requires_grad=True)) + \
            self.fc_x(Variable(x_unroll.type(dtype), requires_grad=False)) + \
            self.fc_y(Variable(y_unroll.type(dtype), requires_grad=False)) + \
            self.fc_r(Variable(r_unroll.type(dtype), requires_grad=False))

        H = torch.tanh(U)
        for i in range(3):
            H = torch.tanh(self.middle[i](H))
        output = torch.sigmoid(self.fc_out(H))

        output = output.view(self.batch_size, self.x_dim, self.y_dim, self.c_dim)

        return output

    def generate(self):
        z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)
        image = self.generator(torch.from_numpy(z))
        image = image.cpu().data.numpy()

        return image
