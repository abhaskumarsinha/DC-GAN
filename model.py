import dataset
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class Model:
    
    input_features = 100
    batch_size = 128
    epochs = 1
    gan = tf.keras.Sequential()
    image_width = 64
    dataset = []
    learning_rate = 0.001
    beta_1 = 0.05
    
    def random_generator_input(self):
        return tf.random.normal((self.batch_size, self.input_features))
    
    def GAN(self):
        
        #generator
        generator = tf.keras.Sequential()
        generator.add(tf.keras.layers.Dense(self.input_features*4*4, input_shape=[self.input_features]))
        generator.add(tf.keras.layers.Reshape([4, 4, self.input_features]))
        generator.add(tf.keras.layers.Conv2DTranspose(int(self.input_features / 2), kernel_size = 5, strides = 2, padding = 'same', activation = 'selu'))
        generator.add(tf.keras.layers.Conv2DTranspose(int(self.input_features / 4), kernel_size = 10, strides = 4, padding = 'same', activation = 'selu'))
        generator.add(tf.keras.layers.Conv2DTranspose(3, kernel_size = 10, strides = 2, padding = 'same', activation = 'sigmoid'))
        generator.summary()
        
        #discriminator
        discriminator = tf.keras.Sequential()
        discriminator.add(tf.keras.layers.Conv2D(200, kernel_size = 4, strides = 1, padding = 'same', input_shape= [self.image_width, self.image_width, 3]))
        discriminator.add(tf.keras.layers.Conv2D(100, kernel_size=5, strides = 2, padding = 'same'))
        discriminator.add(tf.keras.layers.Conv2D(9, kernel_size = 5, strides = 2, padding = 'same'))
        discriminator.add(tf.keras.layers.Flatten())
        discriminator.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
        discriminator.summary()
        
        #GAN
        GAN = tf.keras.Sequential([generator, discriminator])
        self.gan = GAN
        
        #compiling
        
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)
        generator.compile(optimizer=adam_optimizer, loss='binary_crossentropy')
        discriminator.compile(optimizer=adam_optimizer, loss='binary_crossentropy')
        GAN.compile(optimizer=adam_optimizer, loss='binary_crossentropy')
        
        return GAN
    
    def training_steps(self):
        
        generator, discriminator = self.gan.layers[0], self.gan.layers[1]
        
        for epoch in range(self.epochs):
            
            #train Discriminator
            #Real samples = 1, fake samples = 0
            discriminator.trainable = True
            
            generated_samples = generator(tf.random.normal(shape=[self.batch_size, self.input_features]))
            real_samples = self.dataset[epoch * self.batch_size : (epoch + 1) * self.batch_size]
            
            discriminator_input = tf.concat([generated_samples, real_samples], axis = 0)
            
            zeros = tf.zeros(( self.batch_size, 1))
            ones = tf.ones(( self.batch_size, 1))
            
            y_train = tf.concat([zeros, ones], axis = 0)
            
            discriminator.train_on_batch(discriminator_input, y_train)
            
            # Training Generator
            discriminator.trainable = False
            
            input_samples = tf.random.normal(shape = [self.batch_size, self.input_features])
            y_train = tf.ones((self.batch_size, 1))
            
            self.gan.train_on_batch(input_samples, y_train)