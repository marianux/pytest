
# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling1D, Conv1D, Conv2DTranspose
#from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model

from keras.optimizers import RMSprop, Adam
from functools import partial

import keras.backend as K
import tensorflow as tf

import matplotlib.pyplot as plt

import sys

import numpy as np


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        aa = inputs[0]
        weights = K.random_uniform(aa[0,:,:].shape)
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class WGANGP():
    
    def __init__(self, ecg_samp = 2000, leads_generator_idx = [1, 2], lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], learning_rate = 1e-5 ):
        
        self.ecg_samp = ecg_samp
#        self.ecg_leads = len(lead_names)
        self.ecg_leads = 1
        self.channels = 1
        self.k_ui16 = 2**15-1
        self.lead_names = lead_names

        self.leads_generator_idx = leads_generator_idx
        
        self.ecg_shape = (self.ecg_samp, self.ecg_leads)
        
        # del paper de audio : escala de penalidad para el gradiente 
        self.k_lambda = 10
        self.latent_dim = self.ecg_samp // 10
        self.latent_shape = (self.latent_dim, 1)

        self.leads_generator_idx = leads_generator_idx
        
        self.ecg_shape = (self.ecg_samp, self.ecg_leads)
        
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
#        optimizer = RMSprop(lr=0.00005)
        optimizer = Adam(lr=learning_rate, beta_1 = 0.5, beta_2 = 0.9)

        # for debug
        self.first_noise = np.random.uniform(-1, 1, (1, self.latent_dim) )
        self.first_img = 0
        
        '''
        # Build the generator and critic
        self.generator_model = self.build_generator()
        self.critic_model = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator_model.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.ecg_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator_model(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic_model(fake_img)
        valid = self.critic_model(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic_model(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
#        partial_gp_loss = partial(self.gradient_penalty_loss,
#                          averaged_samples=interpolated_img)
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img,
                                  gradient_penalty_weight=self.k_lambda)
        
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer)
#                                        loss_weights=[1, 1, self.k_lambda])
        
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic_model.trainable = False
        self.generator_model.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator_model(z_gen)
        # Discriminator determines validity
        valid = self.critic_model(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        '''
        self.generator_alone = self.build_generator()
        discriminator = self.build_critic()


        # The generator_model is used when we want to train the generator layers.
        # As such, we ensure that the discriminator layers are not trainable.
        # Note that once we compile this model, updating .trainable will have no effect within
        # it. As such, it won't cause problems if we later set discriminator.trainable = True
        # for the discriminator_model, as long as we compile the generator_model first.
        for layer in discriminator.layers:
            layer.trainable = False
        discriminator.trainable = False
        generator_input = Input(shape=(self.latent_dim,))
        generator_layers = self.generator_alone(generator_input)
        discriminator_layers_for_generator = discriminator(generator_layers)
        self.generator_model = Model(inputs=[generator_input],
                                outputs=[discriminator_layers_for_generator])
        # We use the Adam paramaters from Gulrajani et al.
        self.generator_model.compile(optimizer=optimizer,
                                loss=self.wasserstein_loss)
        
        # Now that the generator_model is compiled, we can make the discriminator
        # layers trainable.
        for layer in discriminator.layers:
            layer.trainable = True
        for layer in self.generator_alone.layers:
            layer.trainable = False
        discriminator.trainable = True
        self.generator_alone.trainable = False
        
        # The discriminator_model is more complex. It takes both real image samples and random
        # noise seeds as input. The noise seed is run through the generator model to get
        # generated images. Both real and generated images are then run through the
        # discriminator. Although we could concatenate the real and generated images into a
        # single tensor, we don't (see model compilation for why).
        real_samples = Input(shape=self.ecg_shape)
        generator_input_for_discriminator = Input(shape=(self.latent_dim,))
        generated_samples_for_discriminator = self.generator_alone(generator_input_for_discriminator)
        discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
        discriminator_output_from_real_samples = discriminator(real_samples)
        
        # We also need to generate weighted-averages of real and generated samples,
        # to use for the gradient norm penalty.
        averaged_samples = RandomWeightedAverage()([real_samples,
                                                    generated_samples_for_discriminator])
        # We then run these samples through the discriminator as well. Note that we never
        # really use the discriminator output for these samples - we're only running them to
        # get the gradient norm for the gradient penalty loss.
        averaged_samples_out = discriminator(averaged_samples)
        
        # The gradient penalty loss function requires the input averaged samples to get
        # gradients. However, Keras loss functions can only have two arguments, y_true and
        # y_pred. We get around this by making a partial() of the function with the averaged
        # samples here.
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=self.k_lambda)
        # Functions need names or Keras will throw an error
        partial_gp_loss.__name__ = 'gradient_penalty'
        
        # Keras requires that inputs and outputs have the same number of samples. This is why
        # we didn't concatenate the real samples and generated samples before passing them to
        # the discriminator: If we had, it would create an output with 2 * batch_size samples,
        # while the output of the "averaged" samples for gradient penalty
        # would have only batch_size samples.
        
        # If we don't concatenate the real and generated samples, however, we get three
        # outputs: One of the generated samples, one of the real samples, and one of the
        # averaged samples, all of size batch_size. This works neatly!
        self.critic_model = Model(inputs=[real_samples,
                                            generator_input_for_discriminator],
                                    outputs=[discriminator_output_from_real_samples,
                                             discriminator_output_from_generator,
                                             averaged_samples_out])
        # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both
        # the real and generated samples, and the gradient penalty loss for the averaged samples
        self.critic_model.compile(optimizer=optimizer,
                                    loss=[self.wasserstein_loss,
                                          self.wasserstein_loss,
                                          partial_gp_loss])
#                                    loss=[self.wasserstein_loss,
#                                          self.wasserstein_loss,
#                                          partial_gp_loss])


#    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
#        """
#        Computes gradient penalty based on prediction and weighted real / fake samples
#        """
#        gradients = K.gradients(y_pred, averaged_samples)[0]
#        # compute the euclidean norm by squaring ...
#        gradients_sqr = K.square(gradients)
#        #   ... summing over the rows ...
#        gradients_sqr_sum = K.sum(gradients_sqr,
#                                  axis=np.arange(1, len(gradients_sqr.shape)))
#        #   ... and sqrt
#        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
#        # compute lambda * (1 - ||grad||)^2 still for each single sample
#        gradient_penalty = K.square(1 - gradient_l2_norm)
#        # return the mean as loss over all the batch samples
#        gradient_penalty = K.mean(gradient_penalty)
#        
#        return gradient_penalty
#
#
#    def wasserstein_loss(self, y_true, y_pred):
#        return K.mean(y_true * y_pred)

    def wasserstein_loss(self, y_true, y_pred):
        """Calculates the Wasserstein loss for a sample batch.
        The Wasserstein loss function is very simple to calculate. In a standard GAN, the
        discriminator has a sigmoid output, representing the probability that samples are
        real or generated. In Wasserstein GANs, however, the output is linear with no
        activation function! Instead of being constrained to [0, 1], the discriminator wants
        to make the distance between its output for real and generated samples as
        large as possible.
        The most natural way to achieve this is to label generated samples -1 and real
        samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
        outputs by the labels will give you the loss immediately.
        Note that the nature of this loss means that it can be (and frequently will be)
        less than 0."""
        return K.mean(y_true * y_pred)
    
    
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples, gradient_penalty_weight = 1):
        """Calculates the gradient penalty loss for a batch of "averaged" samples.
        In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
        loss function that penalizes the network if the gradient norm moves away from 1.
        However, it is impossible to evaluate this function at all points in the input
        space. The compromise used in the paper is to choose random points on the lines
        between real and generated samples, and check the gradients at these points. Note
        that it is the gradient w.r.t. the input averaged samples, not the weights of the
        discriminator, that we're penalizing!
        In order to evaluate the gradients, we must first run samples through the generator
        and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
        input averaged samples. The l2 norm and penalty can then be calculated for this
        gradient.
        Note that this loss function requires the original averaged samples as input, but
        Keras only supports passing y_true and y_pred to loss functions. To get around this,
        we make a partial() of the function with the averaged_samples argument, and use that
        for model training."""
        # first get the gradients:
        #   assuming: - that y_pred has dimensions (batch_size, 1)
        #             - averaged_samples has dimensions (batch_size, nbr_features)
        # gradients afterwards has dimension (batch_size, nbr_features), basically
        # a list of nbr_features-dimensional gradient vectors
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)
    
    

    def build_generator(self):
#        with tf.device('/GPU:0'):
        with tf.device('/CPU:0'):

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
            
    
            dim = 2
            dim_mul = 16
            base_dim = int(np.max( [1, self.ecg_samp/(4**3)] ))

            model.add(Dense( base_dim * dim * dim_mul, activation="relu", input_dim=self.latent_dim))
            model.add(Reshape(( base_dim, dim * dim_mul)))
            
            dim_mul //= 2
            
#            model.add(Conv1D(dim * dim_mul, kernel_size=5, dilation_rate=4, padding="same"))
            model.add(Conv1D(dim * dim_mul, kernel_size=25, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))

            dim_mul //= 2
            model.add(Conv1D(dim * dim_mul, kernel_size=25, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))
            
            model.add(Conv1D(1, kernel_size=25, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("tanh"))
            
            model.summary()

            return model
    
    
    def build_critic(self):

#        with tf.device('/GPU:0'):
        with tf.device('/CPU:0'):

            model = Sequential()

#            model.add(Conv1D(128, kernel_size=25, strides=4, input_shape=self.ecg_shape, padding="same"))
            model.add(Conv1D(128, kernel_size=25, strides=4, input_shape=self.ecg_shape, padding="same"))
#            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
#            model.add(Dropout(0.25))
            model.add(Conv1D(64, kernel_size=25, strides=4, padding="same"))
#            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
#            model.add(Dropout(0.25))
            model.add(Conv1D(32, kernel_size=25, strides=4, padding="same"))
#            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
#            model.add(Dropout(0.25))
            model.add(Conv1D(16, kernel_size=25, strides=4, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
#            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(1))
    
            model.summary()
    
#            img = Input(shape=self.ecg_shape)
#            validity = model(img)
#
#        return Model(img, validity)

            return model

    def make_generator():
        """Creates a generator model that takes a 100-dimensional noise vector as a "seed",
        and outputs images of size 28x28x1."""
        model = Sequential()
        model.add(Dense(1024, input_dim=100))
        model.add(LeakyReLU())
        model.add(Dense(128 * 7 * 7))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        if K.image_data_format() == 'channels_first':
            model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
            bn_axis = 1
        else:
            model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
            bn_axis = -1
        model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
        model.add(BatchNormalization(axis=bn_axis))
        model.add(LeakyReLU())
        model.add(Convolution2D(64, (5, 5), padding='same'))
        model.add(BatchNormalization(axis=bn_axis))
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
        model.add(BatchNormalization(axis=bn_axis))
        model.add(LeakyReLU())
        # Because we normalized training inputs to lie in the range [-1, 1],
        # the tanh function should be used for the output of the generator to ensure
        # its output also lies in this range.
        model.add(Convolution2D(1, (5, 5), padding='same', activation='tanh'))
        return model

    def make_discriminator():
        """Creates a discriminator model that takes an image as input and outputs a single
        value, representing whether the input is real or generated. Unlike normal GANs, the
        output is not sigmoid and does not represent a probability! Instead, the output
        should be as large and negative as possible for generated inputs and as large and
        positive as possible for real inputs.
        Note that the improved WGAN paper suggests that BatchNormalization should not be
        used in the discriminator."""
        model = Sequential()
        if K.image_data_format() == 'channels_first':
            model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(1, 28, 28)))
        else:
            model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
        model.add(LeakyReLU())
        model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal',
                                strides=[2, 2]))
        model.add(LeakyReLU())
        model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same',
                                strides=[2, 2]))
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(1024, kernel_initializer='he_normal'))
        model.add(LeakyReLU())
        model.add(Dense(1, kernel_initializer='he_normal'))
        return model


    def train(self, data_gen, epochs, batch_size, sample_interval=50):

        # Load the dataset
#        (X_train, _), (_, _) = mnist.load_data()
#
#        # Rescale -1 to 1
#        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
#        X_train = np.expand_dims(X_train, axis=3)
#
        
        
        '''        
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
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
#                noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                
                # Train the critic
                d_loss = self.critic_model.train_on_batch([X_train, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

        '''
        
        # We make three label vectors for training. positive_y is the label vector for real
        # samples, with value 1. negative_y is the label vector for generated samples, with
        # value -1. The dummy_y vector is passed to the gradient_penalty loss function and
        # is not used.
        positive_y = np.ones((batch_size, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

#        # sample once
#        X_train = next(data_gen)
#        
#        X_train = X_train[:, :, 1:2] / self.k_ui16
        
        for epoch in range(epochs):
            
#            print("Epoch: ", epoch)
#            print("Number of batches: ", int(X_train.shape[0] // batch_size))
            discriminator_loss = []
            generator_loss = []
            
            for j in range(self.n_critic):
                
#                image_batch = discriminator_minibatches[j * batch_size:
#                                                        (j + 1) * batch_size]
#                noise = np.random.rand(batch_size, 100).astype(np.float32)
                
                X_train = next(data_gen)
                
                latent_img = X_train[:, :, 1:2] / self.k_ui16

#                latent_img = X_train + np.random.normal(0, np.sqrt(np.var(X_train)/10), X_train.shape)
                
                # Signal plus noise
#                latent_img = X_train[:, :, self.leads_generator_idx]
#                latent_img =+ np.random.normal(0, np.sqrt(np.var(latent_img)/20), latent_img.shape)                
                
                # just noise
#                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                
                discriminator_loss.append(self.critic_model.train_on_batch(
                                                                        [latent_img, noise],
                                                                        [positive_y, negative_y, dummy_y]))
            generator_loss.append(self.generator_model.train_on_batch(np.random.rand(batch_size, self.latent_dim), positive_y))
        
            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, discriminator_loss[-1][-1], generator_loss[-1]))
        
            if epoch % sample_interval == 0:
                self.sample_images(epoch, latent_img)
                
            if epoch % 100 == 0:
                self.critic_model.save( "checkpoint/critic.h5" )  # creates a HDF5 file 'my_model.h5'
                self.generator_model.save( "checkpoint/generator.h5" )  # creates a HDF5 file 'my_model.h5'
                

    def sample_images(self, epoch, X_train):
        
#        noise = np.random.normal(0, 1, (1, self.latent_dim))
        noise = np.random.uniform(-1, 1, (1, self.latent_dim) )
            
        gen_imgs = self.generator_alone.predict(noise)
        gen_imgs = np.squeeze(gen_imgs)

        first_img = self.generator_alone.predict(self.first_noise)
        first_img = np.squeeze(first_img)

        if epoch == 0:
            self.first_img = self.generator_alone.predict(self.first_noise)
            self.first_img = np.squeeze(self.first_img)

        fig = plt.figure(1)
        plt.cla()
        plt.plot( gen_imgs, label = 'gen' )
        plt.plot( np.squeeze(X_train[0,:]), label = 'real' )
#        plt.plot( first_img, 'g:', label = 'first_noise' )
#        plt.plot( self.first_img, 'r--', label = 'first_img' )
        plt.legend( )
        
        fig.savefig("images/epoch_{:d}.png".format(epoch), dpi=150 )

if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=30000, batch_size=32, sample_interval=100)
