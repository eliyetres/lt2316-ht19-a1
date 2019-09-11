import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from read import read_labels, open_datafiles, match_labels
# Ignore everything past the first 100 characters of each sentence and remove any instances that have less than 100 characters. 


# Requirements: 
# Implement the learning algorithm in PyTorch.

# Use a recurrent neural network variant, such as a Gated Recurrent Unit (GRU) network, be central to the classifiers design. (You will also probably need an output softmax activation layer.)


# Choose any 10 languages and document them in your readme. Split out those languages into smaller training and test files and submit them with your project.


data_dir = "data/" # Data directory

languages = ["Swedish", "Danish", "Bokmål", "Icelandic", "Faroese", "English", "Welsh", "Cornish", "Breton", "Old English "] # Selected languages

lan = read_labels(data_dir, "labels.csv", languages) # Language codes


x_train, y_train = open_datafiles(data_dir, "y_train.txt", "x_train.txt",lan)
x_test, y_test = open_datafiles(data_dir, "y_test.txt", "x_test.txt",lan)

match_labels(data_dir, "labels.csv", "cym")




import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from main import x_train, y_train, x_test, y_test 


batch_size = 10
train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")







parser = argparse.ArgumentParser(description="Gated Recurrent Unit (GRU) networks.")

args = parser.parse_args()