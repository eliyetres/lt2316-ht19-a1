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

    def __init__(self, device, vocab_size, seq_len, input_size, hidden_size, output_size, num_layers=1, dropout=0, lr=0.1, loss_type=2):
        super(GRUNet, self).__init__()
        # Define parameters
        self.device = torch.device(device)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lr = lr
        self.loss_type = loss_type
        # Define layers
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)
        # A linear layer to apply linear transformation to the output features from the RNN module.
        self.linear = nn.Linear(self.hidden_size*seq_len, output_size)

    def set_dev(self, device):
        self.device = device

    def init_model(self, device, vocab_size, seq_len, input_size,  hidden_size, output_size, num_layers=1, dropout=0, lr=0.1, loss_type=2):
        model = GRUNet(device, vocab_size=vocab_size, seq_len=seq_len,  input_size=input_size,
                       hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout, lr=lr, loss_type=loss_type)
        # Defining loss function and optimizer
        # CrossEntropyLoss combines LogSoftmax and NLLLoss in one single class.
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.criterion_mean = nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        return model

    def init_hidden(self, seq_len):
        # An initial pair of hidden weights and memory state.
        return torch.zeros(self.num_layers, seq_len, self.hidden_size).to(self.device)

    def forward(self, X, seq_len):  # X is a batch
        output = self.embedding(X)
        hidden_layer = self.init_hidden(seq_len)
        hidden_layer = hidden_layer.to(self.device)
        # The sentence as indices goes directly into the embedding layer,
        # which selects randomly-initialized vectors corresponding to the
        # indices.
        self.gru.flatten_parameters()  # memory error if I don't use this
        output, hidden_layer = self.gru(output, hidden_layer)

        output = output.contiguous().view(-1, self.hidden_size*len(X[0]))
        output = self.linear(output)
        return output.to(self.device)

    def train_network(self, X_batch, y_batch, model, vocab_size, seq_len, lr=0.1, epochs=20, loss_type=1):
        model.train()
        model = model.to(self.device)
        model.set_dev(self.device)

        print("Training batch...")
        for epoch in range(epochs):
            #print("Epoch: {}".format(epoch+1)
            X_batch = X_batch.to(self.device)   # Push to GPU
            y_batch = y_batch.to(self.device)

            # set the gradients to 0 before backpropagation
            self.optimizer.zero_grad()
            # do the forward pass
            output = model(X_batch, seq_len)

            prefix_len = []
            vocab_len = []
            for prefix in X_batch:
                char_len = torch.nonzero(prefix)  # measure number of chars
                prefix_len.append(char_len.size(0))
                vocab_len.append(len(X_batch[0]))

            prefix_len = torch.FloatTensor(prefix_len)
            prefix_len = prefix_len.to(self.device)
            vocab_len = torch.FloatTensor(vocab_len)
            vocab_len = vocab_len.to(self.device)

            # compute loss
            if loss_type == 1:
                loss = self.criterion_mean(output, y_batch)

            # loss including character prefix length
            if loss_type == 2:
                loss = self.criterion(output, y_batch)
                loss *= (prefix_len/vocab_len)
                loss = loss.mean()

            # additive loss including character prefix
            if loss_type == 3:
                loss = self.criterion(output, y_batch)
                loss += prefix_len
                loss = loss.mean()

            # compute gradients
            loss.backward()

            # update weights
            self.optimizer.step()
            print("Loss {}".format(loss))

        print("Loss: {}".format(loss.item()))
