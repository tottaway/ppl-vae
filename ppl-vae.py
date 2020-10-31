import os

import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from tqdm import tqdm

# for loading and batching MNIST dataset
def setup_data_loaders(batch_size=128, use_cuda=False):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

def sample_image(dataloader):
    dataset = dataloader.dataset
    n_samples = len(dataset)

    # to get a random sample
    random_index = int(np.random.random()*n_samples)
    single_example = dataset[random_index]
    return single_example


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        """Goes from latent space to image"""
        hidden = self.softplus(self.fc1(z))
        loc_img = self.sigmoid(self.fc21(hidden))

        return loc_img

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # unwrap x
        x = x.reshape(-1, 784)

        hidden = self.softplus(self.fc1(x))

        # two output layers/heads
        # one for mean and one for variance
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))

        return z_loc, z_scale

class VAE(nn.Module):
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super().__init__()
        
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))

            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            
            # decode the laten code z
            loc_img = self.decoder(z)
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))

    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def vizualize_reconstruct(self, x):
        print(x[1])
        plt.imshow(x[0][0])
        plt.title("original")
        plt.show()

        z_loc, z_scale = self.encoder(x[0])
        print("z_loc: {}\nz_scale: {}".format(z_loc, z_scale))

        z = dist.Normal(z_loc, z_scale).sample()

        loc_img = self.decoder(z[0])
        loc_img = loc_img.detach().numpy().reshape(28, 28)
        plt.imshow(loc_img)
        plt.title("reconstructed")
        plt.show()

    def traverse_latent_space(self, x1, x2, filepath="./animation.gif"):
        print(x1[1], x2[1])
        z_loc1, z_scale1 = self.encoder(x1[0])
        z_loc2, z_scale2 = self.encoder(x2[0])
        z1 = dist.Normal(z_loc1, z_scale1).sample()
        z2 = dist.Normal(z_loc2, z_scale2).sample()

        fig = plt.figure()
        plt.title("latent space traversal")
        curr_z = z1
        curr_data = self.decoder(z1).detach().numpy().reshape((28, 28))
        im = plt.imshow(curr_data)
        def update(_):
            nonlocal curr_data, curr_z
            curr_z -= 0.02 * z1
            curr_z += 0.02 * z2
            curr_data = self.decoder(curr_z).detach().numpy().reshape((28, 28))
            im.set_array(curr_data)
            return im,
        
        anim = FuncAnimation(fig, update, frames=100, interval=1, blit=True)
        anim.save(filepath, writer='imagemagick', fps=60)

    def reconstruct_img(self, x):
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        loc_img = self.decoder(z)
        return loc_img


def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, _ in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x, _ in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

if __name__ == "__main__":
    # Run options
    LEARNING_RATE = 1.0e-3
    USE_CUDA = False

    # Run only for a single iteration for testing
    NUM_EPOCHS = 10
    TEST_FREQUENCY = 5

    train_loader, test_loader = setup_data_loaders(batch_size=256, use_cuda=USE_CUDA)

    # clear param store
    pyro.clear_param_store()

    # setup the VAE
    vae = VAE(use_cuda = USE_CUDA)

    # setup the optimizer
    adam_args = {"lr": LEARNING_RATE}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in tqdm(range(NUM_EPOCHS)):
        total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % TEST_FREQUENCY == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
            test_elbo.append(-total_epoch_loss_test)
            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

    for i in range(10):
        example1 = sample_image(test_loader)
        example2 = sample_image(test_loader)
        vae.traverse_latent_space(example1, example2,
                filepath="./media/animation{}.gif".format(i))

