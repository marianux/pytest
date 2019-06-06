
# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
#from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K
import tensorflow as tf

import matplotlib.pyplot as plt

import sys

import numpy as np

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        
#        alpha = K.random_uniform( (32, 1, 1, 1) )
        aa = inputs[0]
        alpha = K.random_uniform( aa[0,:,:].shape )
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self, ecg_samp = 2000, leads_generator_idx = [1, 2], lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] ):
        self.ecg_samp = ecg_samp
#        self.ecg_leads = len(lead_names)
        self.ecg_leads = 1
        self.channels = 1
        self.k_ui16 = 2**15-1
        self.lead_names = lead_names

        self.leads_generator_idx = leads_generator_idx
        
        self.ecg_shape = (self.ecg_samp, self.ecg_leads)

        self.latent_dim = 100
        self.latent_shape = (self.latent_dim, 1)

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.ecg_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        with tf.device('/GPU:0'):

            model = Sequential()
    
    #        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
    #        model.add(Reshape((7, 7, 128)))
    #        model.add(UpSampling2D())
    #        model.add(Conv2D(128, kernel_size=4, padding="same"))
    #        model.add(BatchNormalization(momentum=0.8))
    #        model.add(Activation("relu"))
    #        model.add(UpSampling2D())
    #        model.add(Conv2D(64, kernel_size=4, padding="same"))
    #        model.add(BatchNormalization(momentum=0.8))
    #        model.add(Activation("relu"))
    #        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
    #        model.add(Activation("tanh"))

            model.add(Dense(256, activation="relu", input_dim=self.latent_dim))
            model.add(Reshape((16, 16)))
            model.add(UpSampling1D(size=4))
            model.add(Conv1D(4, kernel_size=25, dilation_rate=4, padding="same"))
            model.add(Activation("relu"))
            model.add(UpSampling1D(size=4))
            model.add(Conv1D(2, kernel_size=25, dilation_rate=4, padding="same"))
            model.add(Activation("relu"))
            model.add(UpSampling1D(size=4))
            model.add(Conv1D(1, kernel_size=25, dilation_rate=4, padding="same"))
            model.add(Activation("tanh"))

            model.summary()

            noise = Input(shape=(self.latent_dim,) )
            img = model(noise)

            the_model = Model(noise, img)

        return the_model

    def build_critic(self):

        with tf.device('/GPU:0'):

            model = Sequential()
    
#            model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
#            model.add(LeakyReLU(alpha=0.2))
#            model.add(Dropout(0.25))
#            model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
#            model.add(ZeroPadding2D(padding=((0,1),(0,1))))
#            model.add(BatchNormalization(momentum=0.8))
#            model.add(LeakyReLU(alpha=0.2))
#            model.add(Dropout(0.25))
#            model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
#            model.add(BatchNormalization(momentum=0.8))
#            model.add(LeakyReLU(alpha=0.2))
#            model.add(Dropout(0.25))
#            model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
#            model.add(BatchNormalization(momentum=0.8))
#            model.add(LeakyReLU(alpha=0.2))
#            model.add(Dropout(0.25))
#            model.add(Flatten())
#            model.add(Dense(1))

            model.add(Conv1D(1, kernel_size=25, strides=4, input_shape=self.ecg_shape, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Conv1D(2, kernel_size=25, strides=4, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Conv1D(4, kernel_size=25, strides=4, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Conv1D(8, kernel_size=25, strides=4, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(1))
    
            model.summary()
    
            img = Input(shape=self.ecg_shape)
            validity = model(img)

        return Model(img, validity)

    def train(self, data_gen, epochs, batch_size, sample_interval=50):

        # Load the dataset
#        (X_train, _), (_, _) = mnist.load_data()
#
#        # Rescale -1 to 1
#        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
#        X_train = np.expand_dims(X_train, axis=3)
#
        
        
        
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

#                # Select a random batch of images
#                idx = np.random.randint(0, X_train.shape[0], batch_size)
#                imgs = X_train[idx]
#                # Sample generator input
#                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                
                X_train = next(data_gen)
                
                X_train = X_train[:, :, 1:2] / self.k_ui16

                # Signal plus noise
#                latent_img = X_train[:, :, self.leads_generator_idx]
#                latent_img =+ np.random.normal(0, np.sqrt(np.var(latent_img)/20), latent_img.shape)                
                
                # just noise
                noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                
                # Train the critic
                d_loss = self.critic_model.train_on_batch([X_train, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                
            if epoch % 100 == 0:
                self.critic_model.save( "checkpoint/critic.h5" )  # creates a HDF5 file 'my_model.h5'
                self.generator_model.save( "checkpoint/generator.h5" )  # creates a HDF5 file 'my_model.h5'
                

    def sample_images(self, epoch):
        
        noise = np.random.uniform(-1, 1, (1, self.latent_dim) )
        
        gen_imgs = self.generator.predict(noise)
    
        gen_imgs = np.squeeze(gen_imgs)
    
        fig = plt.figure(1)
        plt.cla()
        plt.plot( gen_imgs, label = 'gen' )
        fig.savefig("images/epoch_{:d}.png".format(epoch) )

if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=30000, batch_size=32, sample_interval=100)
