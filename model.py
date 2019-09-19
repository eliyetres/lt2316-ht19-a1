import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math

""" 
Use a recurrent neural network variant, such as a Gated Recurrent Unit (GRU) network, be central to the classifiers design. (You will also probably need an output softmax activation layer.)
"""

is_cuda = torch.cuda.is_available()

# Change to GPU if available
if is_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


batch_size = 50

train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

class GRUNet(nn.Module):
    """ 
    input_size — The number of expected features in the input x
    hidden_size — The number of features in the hidden state h
    bias — If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    nonlinearity — The non-linearity to use. Default: tanh (Can be tanh or relu.)
    """
    def __init__(self, device, input_size, hidden_size, output_size, n_layers, bidirectional=False, dropout_p = 0, lr=0.01):
        super(GRUNet, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.learning_rate = lr
        self.device = torch.device(device)

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.fc = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.tahn(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device)
        return hidden


    def make_tensor(self, X, Y):
        if self.device == "cpu":
            # Using CPU  
            X = torch.Tensor(X)
            Y = torch.Tensor(Y)
        else:
            # Using GPU                             
            X = torch.as_tensor(X, dtype=torch.float, device=self.device)
            Y = torch.as_tensor(Y, dtype=torch.float, device=self.device)
        return X, Y


    def train(train_loader, learning_rate, epochs=5):

        X, Y = self.make_tensor (X, Y) # Push tensors to GPU
        X = X.to(self.device)
        Y = Y.to(self.device)

        # Defining loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

        for epoch in range(n_epochs):
            print("Epoch: ", epoch)
            # do the forward pass

            # set the gradients to 0 before backpropagation

            # Compute loss

            # compute gradients

            # Update weights

