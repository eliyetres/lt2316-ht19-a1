import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset

class GRUNet(nn.Module):
    """
    Neural Network Module with an embedding layer, a Gated Recurrent Unit (GRU) network module and an output layer
    
    Arguments:
        input_size(int) -- length of the dictionary of embeddings
        embeddings_size(int) -- the size of each embedding vector
        hidden_size(int) -- the number of features in the hidden state 
        output_size(int) -- the number of output classes to be predicted
        num_layers(int, optional) -- Number of recurrent layers. Default=1
        
    Inputs: input_sequence
        input of shape (seq_length, batch_size) -- tensor containing the features 
                                                   of the input sequence
    
    Returns: output
        output of shape (batch_size, output_size) -- tensor containing the sigmoid
                                                     activation on the output features 
                                                     h_t from the last layer of the gru, 
                                                     for the last time-step t.
    """

    def __init__(self, device, vocab_size, batch_size, input_size,  hidden_size, output_size, num_layers=2, dropout=0, lr=0.01):
        super(GRUNet, self).__init__()
        # Define parameters
        self.device = torch.device(device)      
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lr = lr        
        # Define layers
        self.embedding = nn.Embedding(vocab_size, input_size)       
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)        
        # a linear layer to apply linear transformation to the output features from the RNN module. Do i need this?   
        self.linear = nn.Linear(self.hidden_size*batch_size, vocab_size) # hidden layer (?)

    def init_model(self, device,vocab_size, batch_size, input_size,  hidden_size, output_size, num_layers, dropout, lr):
        model = GRUNet(device,vocab_size=vocab_size, batch_size=batch_size,  input_size=input_size,  
                       hidden_size=hidden_size, output_size=output_size, num_layers=num_layers,
                       dropout=dropout, lr=lr)
        # Defining loss function and optimizer
        #criterion = nn.MSELoss()
        #criterion = nn.BCELoss() # none of these
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        return model, criterion, optimizer

    def init_hidden(self, batch_size):
        # An initial pair of hidden weights and memory state.
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

    def shuffle_data(self):
        random.shuffle(self.pairs)
        int_sentences, int_predicted = zip(*self.pairs)
        self.int_sentences = list(int_sentences)
        self.int_predicted = list(int_predicted)
        
    def forward(self, X): # X is a batch
        hidden_layer = self.init_hidden(len(X[0]))
        # The sentence as indices goes directly into the embedding layer,
        # which selects randomly-initialized vectors corresponding to the
        #indices.
        output = self.embedding(X)
        output, hidden = self.gru(output, hidden_layer)
        output = self.linear(output)

        #output = self.softmax(output)
        #out = self.fc(self.tahn(out[:,-1]))
        #out = F.sigmoid(out[-1, :, :])
        return output


    def train(self, X_batch, y_batch, lr=0.1, epochs=20):
        print("Training batch...")
        for epoch in range(epochs):
            print("Epoch: ", epoch+1)

            X_batch, y_batch = self.make_tensor(X_batch, y_batch) # Push tensors to GPU
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)     
        
            # do the forward pass
            self.forward(X_batch)
            # set the gradients to 0 before backpropagation
            self.optimizer.zero_grad()
            # compute loss
            loss = self.cross_entropy_cat() #this is shit
            # compute gradients            
            loss.backward() 
            # update weights
            self.optimizer.step()

        print("Loss: {}".format(loss.item()))
