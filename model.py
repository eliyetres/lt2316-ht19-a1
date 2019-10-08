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
        input of shape (seq_len, batch_size) -- tensor containing the features 
                                                   of the input sequence
    
    Returns: output
        output of shape (seq_len, output_size) -- tensor containing the sigmoid
                                                     activation on the output features 
                                                     h_t from the last layer of the gru, 
                                                     for the last time-step t.
    """

    def __init__(self, device, vocab_size, seq_len, input_size, hidden_size, output_size, num_layers=2, dropout=0, lr=0.01):
        super(GRUNet, self).__init__()
        # Define parameters
        print("Defining parameters...")
        self.device = torch.device(device)      
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lr = lr
        # Define layers
        self.embedding = nn.Embedding(vocab_size, input_size)       
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)        
        # A linear layer to apply linear transformation to the output features from the RNN module.   
        self.linear = nn.Linear(self.hidden_size*seq_len, output_size)

    def init_model(self, device,vocab_size, seq_len, input_size,  hidden_size, output_size, num_layers=2, dropout=0, lr=0.01):
        model = GRUNet(device,vocab_size=vocab_size, seq_len=seq_len,  input_size=input_size,  
                       hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout, lr=lr)
        # Defining loss function and optimizer
        # This criterion combines LogSoftmax and NLLLoss in one single class.
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        return model

    def init_hidden(self, seq_len):
        # An initial pair of hidden weights and memory state.
        return torch.zeros(self.num_layers, seq_len, self.hidden_size).to(self.device)

    def shuffle_data(self):
        random.shuffle(self.pairs)
        int_sentences, int_predicted = zip(*self.pairs)
        self.int_sentences = list(int_sentences)
        self.int_predicted = list(int_predicted)
        
    def forward(self, X): # X is a batch
        output = self.embedding(X)      
        hidden_layer = self.init_hidden(len(X[0]))
        hidden_layer = hidden_layer.to(self.device)
        # The sentence as indices goes directly into the embedding layer,
        # which selects randomly-initialized vectors corresponding to the
        # indices.      
        output, hidden = self.gru(output, hidden_layer)        

        output = output.contiguous().view(-1, self.hidden_size*len(X[0])) # here?

        output = self.linear(output)        
        return output

    #def train(self, X_batch, y_batch, lr=0.01, epochs=20):
    def train(self, dataloader, model, vocab_size, lr=0.01, epochs=20):
        model = model.to(self.device)

        print("Training batch...")
        for epoch in range(epochs):
            print("Epoch: ", epoch+1)
            #for local_batch, local_labels in dataloader:
            for local_batch, local_labels in dataloader:
                #print("labels: ",local_labels)
                #print("data: ", local_batch)
                # Push to GPU
                local_batch = local_batch.to(self.device)   
                local_labels = local_labels.to(self.device)                 
                # set the gradients to 0 before backpropagation
                self.optimizer.zero_grad()                
                # do the forward pass
                output = model(local_batch) # forward
                # compute loss
                loss = self.criterion(output, local_labels)
                # compute gradients            
                loss.backward() 
                # update weights
                self.optimizer.step()
