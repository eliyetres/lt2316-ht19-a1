import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from read import read_labels, open_datafiles, match_labels
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

data_dir = "data/"  # Data directory


# Requirements:
# Implement the learning algorithm in PyTorch.



# Choose any 10 languages and document them in your readme. Split out those languages into smaller training and test files and submit them with your project.
languages = [
    "Swedish", "Danish", "Bokmål", "Icelandic", "Faroese", "English", "Welsh",
    "Cornish", "Breton", "Old English "
]  # Selected languages

lan = read_labels(data_dir, "labels.csv", languages)  # Language codes

x_train, y_train = open_datafiles(data_dir, "y_train.txt", "x_train.txt", lan)
x_test, y_test = open_datafiles(data_dir, "y_test.txt", "x_test.txt", lan)

match_labels(data_dir, "labels.csv", "cym")

batch_size = 10

train_dataset = TensorDataset(torch.from_numpy(x_train),
                              torch.from_numpy(y_train))

test_dataset = TensorDataset(torch.from_numpy(x_test),
                             torch.from_numpy(y_test))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

parser = argparse.ArgumentParser(
    description="Gated Recurrent Unit (GRU) networks.")

args = parser.parse_args()