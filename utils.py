import os
import torch
from torch import nn, optim

def seperate_dataset(data, seperator):
	idx = torch.randperm(data.nelement())
	data_perm = data[idx]
	train_data = data_perm[0:seperator]
	test_data = data_perm[seperator:]
	return train_data, test_data


def save_checkpoint(model, epoch, model_out_path, save_dir, optimizer=None, lr=0.0001, tloss=-1):

	#name = "epoch_{}.pth".format(epoch)
	#model_out_path = os.path.join(save_dir, name)

	if optimizer == None:
		state = {"epoch": epoch ,"model": model,"tloss":tloss}
	else:
		state = {"epoch": epoch ,"model": model,"optimizer":optimizer.state_dict(), "tloss":tloss,
			"lr":lr}
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	torch.save(state, model_out_path)

def load_checkpoint(path):

	if not os.path.exists(path):
		print("Model does not exist")
		return None

	checkpoint = torch.load(path)
	epoch = checkpoint['epoch']
	model = checkpoint['model']
	tloss = checkpoint['tloss']

	optimizer = None
	if 'optimizer' in checkpoint:
		lr = checkpoint['lr']
		optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)
		optimizer.load_state_dict(checkpoint['optimizer'])

	return model, epoch, optimizer, tloss
