import os
import argparse
import torch
import numpy as np
from utils import *
from model import *
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

def train(train_data, model, optimizer):
    model.train()
    train_loss = 0
    l1 = 0
    l2 = 0
    l3 = 0
    l4 = 0
    for batch_idx, y in enumerate(train_data):
        y = y.float()
        y = y.to(device)
        y = y.view(-1,1)
        optimizer.zero_grad()
        l1, l2, l3, l4, total_loss = model.loss_function(y)
        total_loss.backward()
        train_loss += total_loss.item()
        l1 += l1.item()
        l2 += l2.item()
        l3 += l3.item()
        l4 += l4.item()
        optimizer.step()
    # print("avg l1: {}".format(l1))
    # print("avg l2: {}".format(l2))
    # print("avg l3: {}".format(l3))
    # print("avg l4: {}".format(l4))

    return train_loss / train_data.size()[0]



def test(test_data, model):
    test_loss = 0
    with torch.no_grad():
        for batch_idx, y in enumerate(test_data):
            y = y.float()
            y = y.to(device)
            y = y.view(-1,1)
            _, _, _, _, total_loss = model.loss_function(y)
            test_loss += total_loss.item()
    test_loss /= test_data.size()[0]

    return test_loss


def loss_function():
    total_loss = 0
    return total_loss

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='checkpoints',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-p', '--print_every', type=int, default=1,
                        help='how many iterations between print statements')
    parser.add_argument('-t', '--save_interval', type=int, default=1,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    parser.add_argument('-k', '--nr_gaussian_mix', type=int, default=4,
                        help='Number of components in the mixture.')
    parser.add_argument('-z', '--x_dim', type=int, default=2,
                        help='Dimension of the latent variable x.')
    parser.add_argument('-w', '--w_dim', type=int, default=2,
                        help='Dimension of the latent variable w.')
    parser.add_argument('-hd', '--hidden_dim', type=int, default=64,
                        help='Number of neurons of each fc layer.')
    parser.add_argument('-hl', '--hidden_layers', type=int, default=2,
                        help='Number of dense layers in each network.')
    parser.add_argument('-lr', '--lr', type=float,
                        default=0.0001, help='Learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=100,
                        help='Batch size during training per GPU')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=200, help='How many epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    global device
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.load_params:
        model, epoch, _, _ = load_checkpoint(args.load_params)
        print('model parameters loaded')

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if os.path.exists(args.data_dir):
        path = os.path.join(args.data_dir, "traj.npy")
        data = torch.from_numpy(np.load(path))

    train_data, test_data = seperate_dataset(data, 4000000)
    n_samples,input_dim = train_data.size()
    n_samples_test,_ = test_data.size()
    num_batches = int(n_samples / args.batch_size)
    num_batches_test = int(n_samples_test / args.batch_size)
    train_data = train_data[0:num_batches*args.batch_size].view(-1, args.batch_size)
    test_data = test_data[0:num_batches_test*args.batch_size].view(-1, args.batch_size)
    epoch = 0
    best = True
    min_test_loss = 1000000
    train_loss_list = list()
    test_loss_list = list()

    model = GMVAE(K = args.nr_gaussian_mix, sigma = 1, input_dim = input_dim, x_dim = args.x_dim, w_dim = args.w_dim, hidden_dim = args.hidden_dim, hidden_layers = args.hidden_layers, device = device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    while(epoch <= args.max_epochs):

        train_loss = train(train_data, model, optimizer)
        train_loss_list.append(train_loss)
        test_loss = test(test_data, model)
        test_loss_list.append(test_loss)

        if (epoch % args.print_every == 0):
            print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
            print('Epoch: {} Average test loss: {:.4f}'.format(epoch, test_loss))

        if test_loss < min_test_loss:
            best = True

        if (best):
            model_out_path = os.path.join(args.save_dir, "best.pth")
            save_checkpoint(model, epoch, model_out_path, args.save_dir, optimizer, args.lr, -1)
            best = False

        epoch += 1

    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.legend(("train loss", "test loss"), loc="upper right")
    plt.savefig("loss.pdf")


if __name__ == "__main__":
	main()
