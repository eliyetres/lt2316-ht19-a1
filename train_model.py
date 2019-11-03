import argparse
import os
import pickle
import sys
import time
from datetime import datetime 
from collections import Counter
import torch
from torch.utils import data
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

from dataloader import Dataset
from GRUModel import GRUNet
from utils import create_encoding, gen_data, get_vocab, load_data

# Set seed
torch.manual_seed(23)

def train_network_model():
    print("Loading data...")
    X, y, languages = load_data(args.x_file, args.y_file)
    X = X[:-1]
    y = y[:-1]
    vocab = get_vocab(X)
    languages_names = list(languages.keys())
    language_items = dict(Counter(sorted(y)))
    print("Finishing loading data.")
    print("Number of sentences for each language:")
    for lang,no in language_items.items(): 
        print("{}:{:>5}".format(languages_names[lang],no))
    print("Encoding training data...")
    X_encoded = create_encoding(X, vocab)
    print("Finished encoding training data.")

    # Network parameters
    vocab_size = len(vocab) + 1
    input_size = len(X_encoded)
    hidden_size = args.hidden_size
    output_size = len(languages_names)  # number of languages
    seq_len = len(X_encoded[0])
    batch_size = args.batch_size
    num_layers = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device == "cpu":  # is using GPU, pin Dataloader to memory
        pin_memory = False
    else:
        pin_memory = True

    # Generate data
    print("Generating data...")
    X_gen, y_gen = gen_data(X_encoded, y)
    print("Total number of sentences with prefixes: ", len(X_gen))
    training_set = Dataset(X_gen, y_gen)
    training_generator = data.DataLoader(
        training_set, batch_size, pin_memory=pin_memory, shuffle=True)
    print("Finishing processing training data.")

    print("Initilizing network model...")
    model = GRUNet(device, vocab_size=vocab_size, seq_len=seq_len,  input_size=input_size,
                       hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, loss_type=args.loss_type)
    # Defining loss function and optimizer
    # CrossEntropyLoss combines LogSoftmax and NLLLoss in one single class.
    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion_mean = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    
    model.to(device)
    model.train() # initiate training

    print("Training the network...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        print("Epoch {} of {} epochs.".format(epoch, args.epochs))
        for i, (X_batch, y_batch) in enumerate(training_generator):
            if y_batch.size()[0] != batch_size:
                continue
            #print("Batch number {} of {}".format(i+1, len(X_gen)/batch_size))
            #model.train_network(local_batch, local_labels, model, optimizer,criterion,criterion_mean, len(vocab),batch_size, args.lr, (epoch+1), args.loss_type)
   
            #print("Epoch: {}".format(epoch+1)
            X_batch = X_batch.to(device)   # Push to GPU
            y_batch = y_batch.to(device)            
            # do the forward pass
            output = model(X_batch, batch_size)
            output.to(device)
            prefix_len = []
            #vocab_len = []
            # measure number of chars in prefix
            for prefix in X_batch:
                char_len = torch.nonzero(prefix)  
                prefix_len.append(char_len.size(0))
                #vocab_len.append(len(X_batch[0]))
            prefix_len = torch.FloatTensor(prefix_len)
            prefix_len = prefix_len.to(device)
            #vocab_len = torch.FloatTensor(vocab_len)
            #vocab_len = vocab_len.to(device)
            # compute loss
            if args.loss_type == 1:
                loss = criterion_mean(output, y_batch)
            # loss including character prefix length
            if args.loss_type == 2:
                loss = criterion(output, y_batch)
                #loss *= (prefix_len/vocab_len)
                loss *= prefix_len
                loss = loss.mean()
            # additive loss including character prefix
            if args.loss_type == 3:
                loss = criterion(output, y_batch)
                loss += prefix_len
                loss = loss.mean()
            # set the gradients to 0 before backpropagation
            optimizer.zero_grad()
            # calculate avg loss
            epoch_loss += loss.item()
            # compute gradients
            loss.backward()
            # update weights
            optimizer.step()

        print("Loss: {}".format(epoch_loss))

    # Save model to disk
    with open(args.model_name, 'wb+') as tmf:
        pickle.dump(model, tmf)
    # Save vocabulary integer mappings to disk
    with open(args.vocab_name, 'wb+') as f_voc:
        pickle.dump(vocab, f_voc)
    print("Vocabulary integer mappings saved to: {}.".format(args.vocab_name))
    print("A trained model is saved to: {}.".format(args.model_name))


parser = argparse.ArgumentParser(description="Trains the model.")

parser.add_argument("-m", "--model_name", metavar="M", dest="model_name",
                    type=str, help="The name of the network model.")
parser.add_argument("-x", "--x_file", metavar="X", dest="x_file",
                    type=str,  help="File name of the language data.")
parser.add_argument("-y", "--y_file", metavar="Y", dest="y_file",
                    type=str, help="File name of the language labels.")
parser.add_argument("-vo", "--vocab_name", metavar="VO", dest="vocab_name",
                    type=str, help="File name of the vocabulary saved to disk.")
parser.add_argument("-b", "--batch_size", metavar="B", dest="batch_size", type=int,
                    default=100, help="Batch size used for for training the neural network (default 100).")
parser.add_argument("-e", "--epochs", metavar="E", dest="epochs", type=int, default=20,
                    help="Number or epochs used for training the neural network (default 20).")
parser.add_argument("-r", "--lr", metavar="R", dest="lr",
                    type=float, default=0.1, help="Optimizer learning rate (default 0.1).")
parser.add_argument("-l", "--hidden_size", metavar="L", dest="hidden_size",
                    type=int, default=200, help="The size of the hidden layer (default 200).")
parser.add_argument("-t", "--loss_type", metavar="T", dest="loss_type",
                    type=int, default=1, help="Loss type function to use  (default 1).")

args = parser.parse_args()

# Sanity checks
if args.epochs < 0:
    exit("Error: Number of epochs can't be negative.")

if args.hidden_size < 0:
    exit("Error: Size of hidden layer can't be negative.")

if args.lr < 0 or args.lr > 1:
    exit("Error: Learning rate must be a float from 0 and lower than 1, e.g. 0.01.")

if args.loss_type not in [1, 2, 3]:
    exit("Error: Loss types are 1, 2 or 3.")

start_time = datetime.now()
train_network_model()
time_elapsed = datetime.now() - start_time 

print("Time it took to train the model: {}".format(time_elapsed))

#python train_model.py -m small_model_1 -x x_train_small.txt -y y_train_small.txt -vo vocab_small -b 400 -e 80 -r 0.0001 -l 200 -t 1
#python train_model.py -m trained_model_1 -x processed_data/x_train.txt -y processed_data/y_train.txt -vo processed_data/vocab_all -b 500 -e 50 -l 200 -t 1 -r 0.0001
