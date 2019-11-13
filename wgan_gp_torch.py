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
            rel_extrema = train_ds['rel_extrema']
            cant_recordings = len(signals)
    
            # shuffle recordings
            rec_idx = np.random.choice(np.arange(cant_recordings), cant_recordings, replace=False )
        
            for ii in rec_idx:

                train_x = signals[ii]
                train_extrema = rel_extrema[ii]
                cant_samples = train_x.shape[0]
                
                sample_idx = np.random.choice(np.arange(2*dg_sample_size, cant_samples-2*dg_sample_size), batch_size, replace=False )
                
                # trampilla para sync
                
#                tt = [ jj + np.argmax(np.abs(train_x[ jj:jj+dg_sample_size,1]))  for jj in sample_idx ]
#                xx = [ train_x[ jj-hwin_in_samp:jj+hwin_in_samp,:] for jj in tt ]
                
                # xx = [ train_x[ jj-hwin_in_samp:jj+hwin_in_samp,:] for jj in sample_idx ]
                xx = [ np.transpose(train_x[ jj-hwin_in_samp:jj+hwin_in_samp,:]) for jj in sample_idx ]
                rr = [ [ (train_extrema[ll][ np.logical_and(train_extrema[ll] > (jj-hwin_in_samp), train_extrema[ll] < (jj+hwin_in_samp))] - (jj-hwin_in_samp) ) for ll in range(len(train_extrema)) ] for jj in sample_idx ]

                xx = np.stack(xx, axis=0)

                # Get the samples you'll use in this batch
          
                yield ( xx, rr )

os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

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
parser.add_argument("--restore_epoch", type=int, default=0, help="Epoch to restore from a saved model")
parser.add_argument("--restore_lr", type=float, default=0, help="Epoch to restore from a saved model")
opt = parser.parse_args()
print(opt)


cuda = True if torch.cuda.is_available() else False
# cuda = False

ecg_samp = 512
k_ui16 = 2**15-1
leads_generator_idx = [1, 2]

ecg_leads_in = len(leads_generator_idx)

lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

train_samples, train_recordings, train_paths = get_dataset_size(opt.train_list)

train_generator = data_generator(train_paths, opt.batch_size, ecg_samp)

ecg_leads_out = len(lead_names)
imgz_shape = (ecg_samp, ecg_leads_in)
img_shape = (ecg_samp, ecg_leads_out)



class WaveGANGenerator(nn.Module):
    def __init__(self, model_size=64, ngpus=1, num_channels=1, latent_dim=100,
                 post_proc_filt_len=512, verbose=False, upsample=True):
        super(WaveGANGenerator, self).__init__()
        self.ngpus = ngpus
        self.model_size = model_size # d
        self.num_channels = num_channels # c
        self.initial_samp = int(ecg_samp/(np.floor(10**(np.log10(ecg_samp)/5))**5))
        self.latent_dim = latent_dim
        self.post_proc_filt_len = post_proc_filt_len
        self.verbose = verbose

        this_in = latent_dim
        # this_out = int(self.initial_samp**2) * model_size
        # self.fc1 = nn.DataParallel(nn.Linear(this_in, this_out ))
        
        # el view cambia la salida en el forward
        # this_out = self.initial_samp * self.model_size
        this_out = int(ecg_samp * ecg_leads_in / self.initial_samp)
        
        self.tconv1 = None
        self.tconv2 = None
        self.tconv3 = None
        self.tconv4 = None
        self.tconv5 = None
        
                
        self.upSampConv1 = None
        self.upSampConv2 = None
        self.upSampConv3 = None
        self.upSampConv4 = None
        self.upSampConv5 = None
        
        self.upsample = upsample
    
        if self.upsample:
            self.upSampConv1 = nn.DataParallel(
                UpsampleConvLayer(16 * model_size, 8 * model_size, 25, stride=1, upsample=4))
            self.upSampConv2 = nn.DataParallel(
                UpsampleConvLayer(8 * model_size, 4 * model_size, 25, stride=1, upsample=4))
            self.upSampConv3 = nn.DataParallel(
                UpsampleConvLayer(4 * model_size, 2 * model_size, 25, stride=1, upsample=4))
            self.upSampConv4 = nn.DataParallel(
                UpsampleConvLayer(2 * model_size, model_size, 25, stride=1, upsample=4))
            self.upSampConv5 = nn.DataParallel(
                UpsampleConvLayer(model_size, num_channels, 25, stride=1, upsample=4))
            
        else:
            
            this_in = this_out
            this_out =  int(this_in / 2)
            self.tconv1 = nn.ConvTranspose1d(this_in, this_out, 25, stride=4, padding=11, output_padding=1)
            
            this_in = this_out
            this_out =  int(this_in / 2)
            self.tconv2 = nn.ConvTranspose1d(this_in, this_out, 25, stride=4, padding=11, output_padding=1)
            
            this_in = this_out
            this_out =  int(this_in / 2)
            self.tconv3 = nn.ConvTranspose1d(this_in, this_out, 25, stride=4, padding=11, output_padding=1)
            
            this_in = this_out
            this_out =  int(this_in / 2)
            # self.tconv4 = nn.ConvTranspose1d(this_in, this_out, 25, stride=4, padding=11, output_padding=1)
            self.tconv4 = nn.ConvTranspose1d(this_in, num_channels, 25, stride=4, padding=11, output_padding=1)
            
            # this_in = this_out
            # this_out =  int(this_in / 2)
            # self.tconv5 = nn.ConvTranspose1d(this_in, num_channels, 25, stride=4, padding=11, output_padding=1)

        if post_proc_filt_len:
            self.ppfilter1 = nn.Conv1d(num_channels, num_channels, post_proc_filt_len)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):

        if self.verbose:
            print(x.shape)
        # x = self.fc1(x)
        # if self.verbose:
        #     print(x.shape)
        
        x = x.view(-1, int( ecg_samp * ecg_leads_in / self.initial_samp), self.initial_samp)
        # x = F.relu(x)
        output = None
        if self.verbose:
            print(x.shape)
        
        if self.upsample:
            x = F.relu(self.upSampConv1(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.upSampConv2(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.upSampConv3(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.upSampConv4(x))
            if self.verbose:
                print(x.shape)

            output = F.tanh(self.upSampConv5(x))
        else:
            x = F.relu(self.tconv1(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.tconv2(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.tconv3(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.tconv4(x))
            if self.verbose:
                print(x.shape)

            # output = torch.tanh(self.tconv5(x))
            output = torch.tanh(x)
            
        if self.verbose:
            print(output.shape)

        if self.post_proc_filt_len:
            # Pad for "same" filtering
            if (self.post_proc_filt_len % 2) == 0:
                pad_left = self.post_proc_filt_len // 2
                pad_right = pad_left - 1
            else:
                pad_left = (self.post_proc_filt_len - 1) // 2
                pad_right = pad_left
            output = self.ppfilter1(F.pad(output, (pad_left, pad_right)))
            if self.verbose:
                print(output.shape)

        return output



class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary

    If batch shuffle is enabled, only a single shuffle is applied to the entire
    batch, rather than each sample in the batch.
    """

    def __init__(self, shift_factor, batch_shuffle=False):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor
        self.batch_shuffle = batch_shuffle

    def forward(self, x):
        # Return x if phase shift is disabled
        if self.shift_factor == 0:
            return x

        if self.batch_shuffle:
            # Make sure to use PyTorcTrueh to generate number RNG state is all shared
            k = int(torch.Tensor(1).random_(0, 2*self.shift_factor + 1)) - self.shift_factor

            # Return if no phase shift
            if k == 0:
                return x

            # Slice feature dimension
            if k > 0:
                x_trunc = x[:, :, :-k]
                pad = (k, 0)
            else:
                x_trunc = x[:, :, -k:]
                pad = (0, -k)

            # Reflection padding
            x_shuffle = F.pad(x_trunc, pad, mode='reflect')

        else:
            # Generate shifts for each sample in the batch
            k_list = torch.Tensor(x.shape[0]).random_(0, 2*self.shift_factor+1)\
                - self.shift_factor
            k_list = k_list.numpy().astype(int)

            # Combine sample indices into lists so that less shuffle operations
            # need to be performed
            k_map = {}
            for idx, k in enumerate(k_list):
                k = int(k)
                if k not in k_map:
                    k_map[k] = []
                k_map[k].append(idx)

            # Make a copy of x for our output
            x_shuffle = x.clone()

            # Apply shuffle to each sample
            for k, idxs in k_map.items():
                if k > 0:
                    x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k,0), mode='reflect')
                else:
                    x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0,-k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
                                                           x.shape)
        return x_shuffle


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
            reflection_padding = kernel_size // 2
            self.reflection_pad = torch.nn.ConstantPad1d(reflection_padding, value = 0)
#             self.reflection_pad = torch.nn.ReflectionPad1d(reflection_padding)
            self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv1d(out)
        return out


class WaveGANDiscriminator(nn.Module):
    def __init__(self, model_size=64, ngpus=1, num_channels=1, shift_factor=2, alpha=0.2, batch_shuffle=False, verbose=False):
        super(WaveGANDiscriminator, self).__init__()
        self.model_size = model_size # d
        self.ngpus = ngpus
        self.num_channels = num_channels # c
        self.shift_factor = shift_factor # n
        self.alpha = alpha
        self.verbose = verbose
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, etc.)
        self.conv1 = nn.Conv1d(num_channels, model_size, 25, stride=4, padding=11)
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=4, padding=11)
        self.conv3 = nn.Conv1d(2 * model_size, 4 * model_size, 25, stride=4, padding=11)
        self.conv4 = nn.Conv1d(4 * model_size, 8 * model_size, 25, stride=4, padding=11)
        # self.conv5 = nn.Conv1d(8 * model_size, 16 * model_size, 25, stride=4, padding=11)
        self.ps1 = PhaseShuffle(shift_factor, batch_shuffle=batch_shuffle)
        self.ps2 = PhaseShuffle(shift_factor, batch_shuffle=batch_shuffle)
        self.ps3 = PhaseShuffle(shift_factor, batch_shuffle=batch_shuffle)
        self.ps4 = PhaseShuffle(shift_factor, batch_shuffle=batch_shuffle)
        self.fc1 = nn.Linear(16 * model_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        
        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps3(x)

        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        # x = self.ps4(x)

        # x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)

        x = x.view(-1, 16 * self.model_size)
        if self.verbose:
            print(x.shape)

        return torch.sigmoid(self.fc1(x))




def visualize(model, input_size=(20)):
    '''Visualize the input size though the layers of the model'''
    x = torch.randn(input_size).unsqueeze_(dim=0)
    x= x.expand((10,-1))
    
    print('\t-------\n\t|Input|\n\t-------')
    # for layer in list(model.fc1) + list(model.conv1):
    for layer in list(model.model):
        y = layer(x)
        print( '{:s}\n\t|'.format( str(x.size()) ))
        print( '{:s}\n\t|'.format( str(layer)    ))
        x = y
    print('\t-------\n\tOutput\n\t-------\n\t|')
    print( '{:s}\n\t|'.format( str(y.size()) ))


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
            *block(int(np.prod(imgz_shape)), 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
            
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *(ecg_leads_out, ecg_samp))
        return img


class Discriminator_conv(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


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


def plot_examples(fake_imgs, imgs, locations):

    fig = plt.figure(1)
    
    for ii in leads_generator_idx + [3]:
        
        plt.cla()
        
        # plt.plot( np.squeeze(fake_imgs.data[0,:,ii].cpu().detach().numpy()), label = lead_names[ii] + 'gen' )
        # plt.plot( np.squeeze(imgs[0,:,ii]), label = 'real' )
        # plt.plot( locations[0][ii],np.squeeze(fake_imgs.data[0,:,ii].cpu().detach().numpy())[locations[0][ii]], 'rx')
        # plt.plot( locations[0][ii],np.squeeze(imgs[0,:,ii])[locations[0][ii]], 'bo')

        plt.plot( np.squeeze(fake_imgs.data[0,ii,:].cpu().detach().numpy()), label = lead_names[ii] + 'gen' )
        plt.plot( np.squeeze(imgs[0,ii,:]), label = 'real' )
        plt.plot( locations[0][ii],np.squeeze(fake_imgs.data[0,ii,:].cpu().detach().numpy())[locations[0][ii]], 'rx')
        plt.plot( locations[0][ii],np.squeeze(imgs[0,ii,:])[locations[0][ii]], 'bo')
        plt.legend( )
        plt.title(lead_names[ii])
        
        fig.savefig("images/epoch_{:d}_{:s}.png".format(epoch, lead_names[ii]), dpi=150 )


def calc_ecg_values( xx, tt ):

    # yy = [ [ xx[ii][samp_idx, jj] for (jj, samp_idx) in zip(range(len(tt[ii])), tt[ii]) ] for ii in range(xx.shape[0]) ]
    yy = [ [ xx[ii][jj, samp_idx] for (jj, samp_idx) in zip(range(len(tt[ii])), tt[ii]) ] for ii in range(xx.shape[0]) ]

    return(yy)


def my_mse_loss_func( xx, yy):

    mse_loss_func = nn.MSELoss()

    my_mse = torch.mean(torch.stack([ torch.mean(torch.stack([ mse_loss_func(xxx,yyy) for (xxx,yyy) in zip(xx[ii],yy[ii]) ])) for ii in range(len(xx)) ]))

    return(my_mse)


# Loss weight for gradient penalty
lambda_gp = 10


# Initialize generator and discriminator
bVerbose = False
# bVerbose = True

generator = WaveGANGenerator(model_size=64, ngpus=1, num_channels=ecg_leads_out, verbose=bVerbose, 
                              latent_dim=int(np.prod(imgz_shape)), post_proc_filt_len=512, upsample=False)

discriminator = WaveGANDiscriminator( model_size=64, ngpus=1, num_channels=ecg_leads_out, verbose=bVerbose, 
                                      shift_factor=2, alpha=0.2, batch_shuffle=False)


# generator = Generator()
# discriminator = Discriminator()


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

if( opt.restore_epoch > 0 ):
    
    if( opt.restore_lr == 0 ):
        # same value        
        opt.restore_lr = opt.lr 

    generator.load_state_dict( torch.load("models/generator_{:f}_{:d}.trc".format(opt.restore_lr, opt.restore_epoch)) )
    # discriminator.load_state_dict( torch.load("models/discriminator_{:f}_{:d}.trc".format(opt.restore_lr, opt.restore_epoch)) )

    print( 'Model restored @ epoch {:d} - lr:{:f}'.format(opt.restore_epoch, opt.restore_lr) )

# ----------
#  Training
# ----------


batches_done = 0
losses = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for epoch in range(opt.restore_epoch, opt.n_epochs):
#    for i, (imgs, _) in enumerate(dataloader):

    (imgs, rr ) = next(train_generator)
    
    imgs = imgs / k_ui16
    # imgs_z = imgs[:, :, leads_generator_idx]
    imgs_z = imgs[:, leads_generator_idx,:]
    imgs_z = imgs_z + np.random.normal(0, np.sqrt(np.var(imgs_z)/20), imgs_z.shape)
    imgs_z = imgs_z.reshape( (imgs_z.shape[0], -1) , order='F')

    # Configure input
    real_imgs = Variable((torch.from_numpy(imgs)).type(Tensor))
    
    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Sample noise as generator input
#    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
#    z = Variable(Tensor(np.random.uniform(-1, 1, (imgs.shape[0], opt.latent_dim))))
    z = Variable((torch.from_numpy(imgs_z)).type(Tensor))

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

        mse_loss = my_mse_loss_func( calc_ecg_values(real_imgs, rr) , calc_ecg_values(fake_imgs, rr) )

        g_loss = -torch.mean(fake_validity) + mse_loss

        g_loss.backward()
        optimizer_G.step()

        print(
            "[Epoch %d/%d] [D loss: %f] [Real val: %f] [Fake val: %f] [Lambda val: %f] || [G loss: %f] [MSE loss: %f]"
            % (epoch, opt.n_epochs, d_loss.item(), -torch.mean(real_validity), torch.mean(fake_validity), lambda_gp * gradient_penalty, g_loss.item(), mse_loss, )
        )
        
        losses = np.vstack(( losses, ( epoch, d_loss.item(), -torch.mean(fake_validity), mse_loss, -torch.mean(real_validity), torch.mean(fake_validity),  lambda_gp * gradient_penalty )) )
        
        if batches_done % opt.sample_interval == 0:

            plot_examples(fake_imgs, imgs, rr)
            
            fig = plt.figure(1)
            
            plt.cla()
                
            plt.plot( np.vstack(losses)[1:,0], np.vstack(losses)[1:,1], label = 'disc' )
            plt.plot( np.vstack(losses)[1:,0], np.vstack(losses)[1:,2], label = 'gen' )
            plt.plot( np.vstack(losses)[1:,0], np.vstack(losses)[1:,3], label = 'mse' )
            plt.legend( )
            plt.xlabel('epochs')
            plt.title('Losses')
            
            fig.savefig("images/losses.png", dpi=150 )
            
        batches_done += opt.n_critic

    if epoch % (30 * opt.sample_interval) == 0:

        this_backup = "images/{:d}".format(epoch)
        os.makedirs(this_backup, exist_ok=True)  

        os.system('mv images/epoch_*.png {:s}'.format(this_backup) )
        
    if epoch % (100 * opt.sample_interval) == 0:
        
        torch.save(generator.state_dict(), "models/generator_{:f}_{:d}.trc".format(opt.lr, epoch))
        torch.save(discriminator.state_dict(), "models/discriminator_{:f}_{:d}.trc".format(opt.lr, epoch))


# python wgan_gp_torch.py --train_list /media/datasets/gan_tests/train_size.txt --n_epochs 10000000 --batch_size 32 --lr 0.0001 --sample_interval 10