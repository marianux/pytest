import argparse
import os
import numpy as np
import math
import sys
from glob import glob
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt

#import torchvision.transforms as transforms
#from torchvision.utils import save_image

from torch.utils.data import DataLoader
#from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

def get_dataset_size(train_list_fn):

    train_samples = 0.0
    
    if os.path.isfile( train_list_fn ):
        
        aux_df = []
        
        base_path, _ = os.path.split(train_list_fn)
        
        try:
            aux_df = read_csv(train_list_fn, header=None, index_col=False, sep=',' )

            paths = aux_df[0].values
            paths = paths.tolist()
            
            paths = [ os.path.join(base_path, each_ds) for each_ds in  paths]
            
            train_samples = np.sum( aux_df[1].values )
            train_recordings = np.sum( aux_df[2].values )
            
        except:
            
            try:
                paths = np.loadtxt(train_list_fn, dtype=list ).tolist()
            except:
                paths = glob(train_list_fn)
            
    else:
        paths = glob(train_list_fn)
    
    if not isinstance(paths, list):
        paths = [paths]
            
    return train_samples, train_recordings, paths


def data_generator( datasets, batch_size, dg_sample_size):
    """A generator yields (source, target) arrays for training."""

    hwin_in_samp = dg_sample_size//2

    while True:
          
        cant_ds = len(datasets)
    
        # Shuffle datasets
        datasets = np.random.choice(datasets, cant_ds, replace=False )
        
        for ds_idx in range(cant_ds):
            
            this_ds = datasets[ds_idx]
#            print('\nEntering:' + this_ds + '\n')
            train_ds = np.load(this_ds)[()]
            signals = train_ds['signals']
            cant_recordings = len(signals)
    
            # shuffle recordings
            rec_idx = np.random.choice(np.arange(cant_recordings), cant_recordings, replace=False )
        
            for ii in rec_idx:

                train_x = signals[ii]
                cant_samples = train_x.shape[0]
                
                sample_idx = np.random.choice(np.arange(2*dg_sample_size, cant_samples-2*dg_sample_size), batch_size, replace=False )
                
                # trampilla para sync
                
                tt = [ jj + np.argmax(np.abs(train_x[ jj:jj+dg_sample_size,1]))  for jj in sample_idx ]
                
                xx = [ train_x[ jj-hwin_in_samp:jj+hwin_in_samp,:] for jj in tt ]
                
                xx = np.stack(xx, axis=0)

                # Get the samples you'll use in this batch
          
                yield ( xx )

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument( '--train_list', 
                     default='', 
                     type=str, 
                     help='Nombre de la base de datos')

parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)


cuda = True if torch.cuda.is_available() else False
#cuda = False

ecg_samp = 128
ecg_leads = 1
k_ui16 = 2**15-1

train_samples, train_recordings, train_paths = get_dataset_size(opt.train_list)

train_generator = data_generator(train_paths, opt.batch_size, ecg_samp)

img_shape = (ecg_samp, ecg_leads)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()


if cuda:
    generator.cuda()
    discriminator.cuda()

## Configure data loader
#os.makedirs("../../data/mnist", exist_ok=True)
#dataloader = torch.utils.data.DataLoader(
#    datasets.MNIST(
#        "../../data/mnist",
#        train=True,
#        download=True,
#        transform=transforms.Compose(
#            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#        ),
#    ),
#    batch_size=opt.batch_size,
#    shuffle=True,
#)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
#    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = Tensor(np.random.random(real_samples.shape))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty




# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):
#    for i, (imgs, _) in enumerate(dataloader):

    imgs = next(train_generator)
    
    imgs = imgs[:, :, 1:2] / k_ui16

    # Configure input
    real_imgs = Variable((torch.from_numpy(imgs)).type(Tensor))

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

    # Generate a batch of images
    fake_imgs = generator(z)

    # Real images
    real_validity = discriminator(real_imgs)
    # Fake images
    fake_validity = discriminator(fake_imgs)
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
    # Adversarial loss
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

    d_loss.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()

    # Train the generator every n_critic steps
    if epoch % opt.n_critic == 0:

        # -----------------
        #  Train Generator
        # -----------------

        # Generate a batch of images
        fake_imgs = generator(z)
        # Loss measures generator's ability to fool the discriminator
        # Train on fake images
        fake_validity = discriminator(fake_imgs)
        g_loss = -torch.mean(fake_validity)

        g_loss.backward()
        optimizer_G.step()

        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, d_loss.item(), g_loss.item())
        )

        if batches_done % opt.sample_interval == 0:
            fig = plt.figure(1)
            plt.cla()
            plt.plot( np.squeeze(fake_imgs.data[0,:,:].cpu().detach().numpy()), label = 'gen' )
            plt.plot( np.squeeze(imgs[0,:,:]), label = 'real' )
            plt.legend( )
            
            fig.savefig("images/epoch_{:d}.png".format(epoch), dpi=150 )

        batches_done += opt.n_critic

