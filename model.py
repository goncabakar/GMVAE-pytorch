import torch
from torch import nn, optim
from torch.nn import functional as F
from fcnet import *
import math

class GMVAE(nn.Module):
    def __init__(self, K, sigma, input_dim, x_dim, w_dim, hidden_dim, hidden_layers, device):

        super(GMVAE, self).__init__()

        self.K = K
        self.sigma = sigma
        self.input_dim = input_dim
        self.x_dim = x_dim
        self.w_dim = w_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.device = device

        # Q_xw
        self.fc_q1 = FCNet(input_dim = self.input_dim, output_dim = self.hidden_dim, hidden_dim = self.hidden_dim, hidden_layers = self.hidden_layers, act_out = "relu")
        self.fc_mean_x = nn.Linear(self.hidden_dim, self.x_dim)
        self.fc_var_x = nn.Linear(self.hidden_dim, self.x_dim)
        self.fc_mean_w = nn.Linear(self.hidden_dim, self.w_dim)
        self.fc_var_w = nn.Linear(self.hidden_dim, self.w_dim)
        self.fc_qz = nn.Linear(self.hidden_dim, self.K)

        # Px_wz
        self.fc_x_wz = FCNet(input_dim = self.w_dim, output_dim = self.hidden_dim, hidden_dim = self.hidden_dim, hidden_layers = 0, act_out = "relu")
        self.fc_x_means = nn.ModuleList()
        self.fc_x_vars = nn.ModuleList()
        self.x_mean_list = list()
        self.x_var_list = list()

        # Py_x
        self.fc_pyx = FCNet(input_dim = self.x_dim, output_dim = self.input_dim*2, hidden_dim = self.hidden_dim, hidden_layers = self.hidden_layers, act_out = None)
        for i in range(self.K):
            self.fc_x_means.append(nn.Linear(self.hidden_dim, self.x_dim))
            self.fc_x_vars.append(nn.Linear(self.hidden_dim, self.x_dim))


    def Q_xw(self, y):

        h = self.fc_q1(y)
        mean_x = self.fc_mean_x(h)
        var_x = torch.exp(self.fc_var_x(h))
        mean_w = self.fc_mean_w(h)
        var_w = torch.exp(self.fc_var_w(h))
        qz = F.softmax(self.fc_qz(h), dim=1)

        return qz, mean_x, var_x, mean_w, var_w

    def Px_wz(self, w):
        h2 = self.fc_x_wz(w)
        self.x_mean_list = []
        self.x_var_list = []


        for i, l in enumerate(self.fc_x_means):
            self.x_mean_list.append(l(h2))
        for i, l in enumerate(self.fc_x_vars):
            a = l(h2)
            self.x_var_list.append(torch.exp(l(h2)))

        return self.x_mean_list, self.x_var_list


    def Py_x(self, x):
        params = self.fc_pyx(x)
        mean_y = params[:, 0:self.input_dim]
        var_y = torch.exp(params[:, self.input_dim:])

        return mean_y, var_y

    def reparameterize(self, mu, var, dim1, dim2):
        eps = torch.randn(dim1, dim2).to(self.device)
        return mu + eps*torch.sqrt(var)

    def loss_function(self, y):
        qz, mean_x, var_x, mean_w, var_w = self.Q_xw(y)
        w_sample = self.reparameterize(mu = mean_w, var = var_w, dim1 = mean_w.size()[0], dim2 = mean_w.size()[1])
        x_sample = self.reparameterize(mu = mean_w, var = var_w, dim1 = mean_x.size()[0], dim2 = mean_x.size()[1])
        y_recons_mean, y_recons_var = self.Py_x(x_sample)
        x_mean_list, x_var_list = self.Px_wz(w_sample)
        y_recons = self.reparameterize(mu = y_recons_mean, var = y_recons_var, dim1 = y_recons_mean.size()[0], dim2 = y_recons_mean.size()[1])
        recons_loss = torch.sum(torch.pow(y - y_recons, 2),0)
        recons_loss = recons_loss / y.size()[0]
        reg_w_loss = self.KL_gaussian_loss(mean_w, var_w)
        reg_z_loss = self.KL_uniform_loss(qz)
        reg_cond_loss = self.KL_conditional_loss(qz, mean_x, var_x, x_mean_list, x_var_list)
        total_loss = recons_loss + reg_w_loss + reg_z_loss + reg_cond_loss

        return recons_loss, reg_w_loss, reg_z_loss, reg_cond_loss, total_loss

    def KL_gaussian_loss(self, mean, var):
        logvar = torch.log(var)
        loss = 0.5 * torch.sum(torch.sum(var + torch.pow(mean, 2) - 1  - logvar, 1), 0)

        return loss / (mean.size()[0]*mean.size()[1])

    def KL_uniform_loss(self, qz):
        loss = 0
        for k in range(self.K):
            loss = loss + torch.sum(qz[:,k] * (torch.log(self.K * qz[:,k] + 1e-10)),0)

        return loss / (qz.size()[0]*qz.size()[1])

    def KL_conditional_loss(self, qz, mean_x, var_x, x_mean_list, x_var_list):
        # KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 )
        x_mean_stack = torch.stack(x_mean_list)
        x_var_stack = torch.stack(x_var_list)
        K, bs, num_sample = x_mean_stack.size()
        loss = 0
        for i in range(num_sample):
            x_mean_2 = x_mean_stack[:,:,i].view(bs, K)
            x_mean_1 = mean_x[:,i].view(bs, -1).repeat(1, K)
            x_var_2 = x_var_stack[:,:,i].view(bs, K)
            x_var_1 = var_x[:,i].view(bs, -1).repeat(1, K)
            KL_batch = 0.5 * (torch.log(x_var_2) - torch.log(x_var_1) - 1 + (x_var_1 + torch.pow(x_mean_1 - x_mean_2, 2))/x_var_2)
            weighted_KL = torch.sum(KL_batch*qz, 1)
            loss = loss + torch.sum(weighted_KL,0)/weighted_KL.size()[0]

        return loss / num_sample


        return 0
