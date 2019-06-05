from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import tensorflow as tf

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

class WGAN():
    def __init__(self, ecg_samp = 2000, leads_generator_idx = [1, 2], lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] ):
        
        self.ecg_samp = ecg_samp
        self.ecg_leads = len(lead_names)
        self.channels = 1
        self.k_ui16 = 2**15-1
        self.lead_names = lead_names
        
        self.leads_generator_idx = leads_generator_idx
        
        self.ecg_shape = (self.ecg_samp, self.ecg_leads, 1)
        self.latent_shape = (self.ecg_samp, len(self.leads_generator_idx), 1)
        self.latent_dim = self.ecg_samp * len(self.leads_generator_idx)

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        print('Build GAN critic')
        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        print('Build GAN generator')
        
        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape= self.latent_shape )
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        the_model = []

        with tf.device('/GPU:0'):
                
            model = Sequential()
    
    #        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim ))
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
    
            model.add(Conv2D( 64 , kernel_size=2, padding="same", input_shape = self.latent_shape ))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))
            model.add(Conv2D( 32 , kernel_size=2, padding="same", input_shape = self.latent_shape ))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))
            model.add(Conv2D(16, kernel_size=4, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))
            model.add(Conv2D(8, kernel_size=4, padding="same"))
            model.add(Dense(self.ecg_leads // 2, activation="tanh"))
            model.add(Reshape((self.ecg_samp, self.ecg_leads, 1)))
    
            model.summary()
    
            noise = Input(shape=self.latent_shape )
            img = model(noise)

            the_model = Model(noise, img)

        return the_model

    def build_critic(self):

        the_model = []
        
        with tf.device('/GPU:0'):
        
            model = Sequential()
    
    #        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.ecg_shape, padding="same"))
    #        model.add(LeakyReLU(alpha=0.2))
    #        model.add(Dropout(0.25))
    #        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    #        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    #        model.add(BatchNormalization(momentum=0.8))
    #        model.add(LeakyReLU(alpha=0.2))
    #        model.add(Dropout(0.25))
    #        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    #        model.add(BatchNormalization(momentum=0.8))
    #        model.add(LeakyReLU(alpha=0.2))
    #        model.add(Dropout(0.25))
    #        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    #        model.add(BatchNormalization(momentum=0.8))
    #        model.add(LeakyReLU(alpha=0.2))
    #        model.add(Dropout(0.25))
    #        model.add(Flatten())
    #        model.add(Dense(1))
    
            model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.ecg_shape, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
            model.add(ZeroPadding2D(padding=((0,1),(0,1))))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(1))
    
            model.summary()
    
            img = Input(shape=self.ecg_shape)
            validity = model(img)

            the_model = Model(img, validity)
        
        return the_model

    def train(self, data_gen, epochs = 100, batch_size=128, sample_interval=50):

#        # Load the dataset
#        (X_train, _), (_, _) = mnist.load_data()
#
#        # Rescale -1 to 1
#        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
#        X_train = np.expand_dims(X_train, axis=3)

        
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

#                # Select a random batch of images
#                idx = np.random.randint(0, X_train.shape[0], batch_size)
#                imgs = X_train[idx]
#                
#                # Sample noise as generator input
#                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                X_train = next(data_gen)
                
                X_train = X_train / self.k_ui16
                
                latent_img = X_train[:, :, self.leads_generator_idx]
                
                latent_img =+ np.random.normal(0, np.sqrt(np.var(latent_img)/20), latent_img.shape)                
                
                latent_img = np.expand_dims(latent_img, axis=3)
                X_train = np.expand_dims(X_train, axis=3)

                # Generate a batch of new images
                gen_imgs = self.generator.predict(latent_img)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(X_train, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(latent_img, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, latent_img, X_train)
                
            if epoch % 100 == 0:
                self.combined.save( "checkpoint/combined.h5" )  # creates a HDF5 file 'my_model.h5'
                self.critic.save( "checkpoint/critic.h5" )  # creates a HDF5 file 'my_model.h5'
                self.generator.save( "checkpoint/generator.h5" )  # creates a HDF5 file 'my_model.h5'
                

    def sample_images(self, epoch, latent_img, X_train):
        
        gen_imgs = self.generator.predict(latent_img[0:1,:,:,:])

        real_imgs = np.squeeze(X_train[0,:,:,:])
        # Rescale images 0 - 1
        gen_imgs = np.squeeze(gen_imgs)
    
        for ii in range(real_imgs.shape[1]):
        
            fig = plt.figure(1)
            plt.cla()
            plt.plot( gen_imgs[:,ii], label = 'gen' )
            plt.plot( real_imgs[:,ii], label = 'real' )
            plt.title( 'Lead {:s}'.format(self.lead_names[ii]) )
            fig.savefig("images/{:d}_lead_{:s}.png".format(epoch, self.lead_names[ii] ) )

#            ii = 0; fig = plt.figure(2); plt.plot( real_imgs[:,ii] ); plt.title( 'Lead {:s}'.format(self.lead_names[ii]) ); plt.show();


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=4000, batch_size=32, sample_interval=50)
