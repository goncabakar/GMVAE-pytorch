from fcnet import *
import torch
from model import *

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

y = torch.randn(100, 1).to(device)
# lol = FCNet(input_dim = 10, output_dim = 5, hidden_dim = 8, hidden_layers = 2, act_out = "relu").to(device)
# x = lol(y)
# print("Dimension of x is : {}".format(x.size()))

model = GMVAE(K = 4, sigma = 1, input_dim = 1, x_dim = 2, w_dim = 2, hidden_dim = 64, hidden_layers = 2, device = device)

model = model.to(device)

qz, mean_x, var_x, mean_w, var_w = model.Q_xw(y)
#print(mean_x.type())

# print("Size of mean_x is : {}".format(mean_x.size()))
print("Size of qz is : {}".format(qz.size()))
# print("Mean of w is : {}".format(mean_w))
# print("Var of w is : {}".format(var_w))
# print("Qz is : {}".format(qz))

w_sample = model.reparameterize(mu = mean_w, var = var_w, dim1 = mean_w.size()[0], dim2 = mean_w.size()[1])
#x_sample = model.reparameterize(mu = mean_w, var = var_w, dim1 = mean_x.size()[0], dim2 = mean_x.size()[1])

#dist = model.reparameterize(mu = mean_x, var = var_x, dim1 = 3, dim2 = 3, device = device)

x_mean_list, x_var_list = model.Px_wz(w_sample)

print("size of 1st element of x_mean_list is : {}".format(x_mean_list[0].size()))
#print("type of 1st element of x_mean_list is : {}".format(x_mean_list[0].type()))
print("length of x_mean_list is : {}".format(len(x_mean_list)))



x_sample_mixture = torch.Tensor(x_mean_list[0].size()).to(device)
for i in range(len(x_mean_list)):
    dim1 = x_mean_list[0].size()[0]
    dim2 = x_mean_list[0].size()[1]
    x_sample_mixture = x_sample_mixture + model.reparameterize(mu = x_mean_list[i], var = x_var_list[i], dim1 = dim1, dim2 = dim2)
    #print(qz[:,i].size())
    #print(x_sample.size())

def KL_conditional_loss(qz, mean_x, var_x, x_mean_list, x_var_list):
    x_mean_stack = torch.stack(x_mean_list)
    x_var_stack = torch.stack(x_var_list)
    K, bs, num_sample = x_mean_stack.size()
    loss = 0
    for i in range(num_sample):
        x_mean_2 = x_mean_stack[:,:,i].view(bs, K)
        x_mean_1 = mean_x[:,i].view(bs, -1).repeat(1, K)
        x_var_2 = x_var_stack[:,:,i].view(bs, K)
        x_var_1 = var_x[:,i].view(bs, -1).repeat(1, K)
        # KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 )
        KL_batch = 0.5 * (torch.log(x_var_2) - torch.log(x_var_1) - 1 + (x_var_1 + torch.pow(x_mean_1 - x_mean_2, 2))/x_var_2)
        weighted_KL = torch.sum(KL_batch*qz, 1)
        loss = loss + torch.sum(weighted_KL,0)/weighted_KL.size()[0]

    return loss / num_sample

lol = KL_conditional_loss(qz, mean_x, var_x, x_mean_list, x_var_list)
#print("x_var_list is : {}".format(x_var_list))

#y_recons_mean, y_recons_var = model.Py_x(x_sample)
#y_recons = model.reparameterize(mu = y_recons_mean, var = y_recons_var, dim1 = y_recons_mean.size()[0], dim2 = y_recons_mean.size()[1])

#print("Size of reconstructed y {}".format(y_recons.size()))
