import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.mixture import GaussianMixture
import scipy
from utils import *
from model import *

# args.cuda = not args.no_cuda and torch.cuda.is_available()
# global device
# device = torch.device("cuda" if args.cuda else "cpu")
path_params = "./checkpoints/best.pth"
model, epoch, _, _ = load_checkpoint(path_params)

data = np.load("./data/traj.npy")

dim,_ = data.shape

n_bins = 100
plt.figure()
n, bins, patches = plt.hist(data, bins=n_bins, density=True, histtype='bar', facecolor='g')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

num_comp = 4
gmm = GaussianMixture(n_components = num_comp)
gmm.fit(data)
means = gmm.means_
covar = gmm.covariances_
probs = gmm.predict_proba(data)
labels = gmm.predict(data)
# print(labels.max(), labels.min())
# np.save("./data/labels.npy", labels)
means = np.array(means)
print(means)
means.sort(axis=1)
print(means)
idx_list = np.zeros(num_comp)


for i in range(num_comp):
    idx_list[i] = np.argmax((sorted[i] == means).astype(int))
print(idx_list)

# for i in range(num_comp):
#     num = sum(probs[:,i])
#     mu = means[i]
#     sigma = np.sqrt(covar[i])
#     y = scipy.stats.norm.pdf(bins, mu, sigma)
#     l = plt.plot(bins, np.transpose(y)*num/dim, 'r--', linewidth=1)


plt.savefig("histogram"+str(num_comp)+".pdf")
