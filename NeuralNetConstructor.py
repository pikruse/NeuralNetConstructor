from tensorflow.keras.layers import Input, Conv1D, Conv2D, Flatten, Dense, Conv1DTranspose, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model



import numpy as np
from numpy.random import randn
import pandas as pd
import matplotlib.pyplot as plt

# define ANN class
class ANN():
    
    # initialize attributes
    def __init__(self,
                 input_dim,
                 neurons,
                 activation,
                 output_activation,
                 output_neurons,
                 batch_size,
                 epochs,
                 early_stopping=False,
                 use_batch_norm=False,
                 use_dropout=False,
                 dropout_rate=None):
        
        self.name = 'artificial_neural_network'
        self.input_dim = input_dim
        self.neurons = neurons #list input - length of list is number of hidden layers
        self.activation = activation
        self.output_activation = output_activation
        self.output_neurons = output_neurons
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.early_stopping = early_stopping
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        
        # derive number of layers from length of neuron list
        self.n_layers = len(neurons)
        
        # call _build method to create network
        self._build()
        
    # build the network architecture
    def _build(self): 
        
        # define input
        inp = Input(shape=self.input_dim, name="input")
        x = inp
        
        #normalize initial input
        if self.use_batch_norm:
                x = BatchNormalization()(x)
        
        # create layers using for loop
        for i in range(self.n_layers):
            
            # make dense layer
            dense_layer = Dense(self.neurons[i], activation = self.activation, name="layer_"+str(i))
            x = dense_layer(x)
            
            # add dropout
            if self.use_dropout:
                x = Dropout(self.dropout_rate)(x)
            
            # add batch norm
            if self.use_batch_norm:
                x = BatchNormalization()(x)
        
        # output layer
        output = Dense(self.output_neurons, activation=self.output_activation)(x)
        
        # create model
        self.model = Model(inp, output)
    
    # compile the model 
    def compile(self):
        
        if self.output_neurons == 1:
            model_loss = 'binary_crossentropy'
        else:
            model_loss = 'categorical_crossentropy'

        
        # compile model
        self.model.compile(optimizer='Adam',
                           loss=model_loss,
                           metrics=['AUC', 'accuracy'])
        
    def train(self, x_train, y_train):
        
        # get batch size and number of epochs
        batch_size = self.batch_size
        epochs = self.epochs
        
        # train model
        self.model.fit(x_train, y_train,
                       epochs = epochs,
                       batch_size = batch_size,
                       validation_split = .2, 
                       verbose = 0)
        
        # evaluate training performance
        train_loss, train_auc, train_acc = self.model.evaluate(x_train, y_train)
        print("Train Loss: {:.4f} \n Train AUC: {:.4f} \n Train Accuracy: {:.4%}".format(train_loss, train_auc, train_acc))
        
        
    def test(self, x_test, y_test):
        # evaluate testing performance
        test_loss, test_auc, test_acc = self.model.evaluate(x_test, y_test)
        print("Test Loss: {:.4f} \n Test AUC: {:.4f} \n Test Accuracy: {:.4%}".format(test_loss, test_auc, test_acc))
    
            
            
            
        
        
        