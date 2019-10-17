import argparse
import os
import pickle
import sys
import time

import torch
from torch.utils import data

from dataloader import Dataset
from get_data import (create_encoding, gen_data, get_vocab, load_data)
from model import GRUNet
from time import strftime
from time import gmtime

def train_model():
    print("Loading data...")
    #X, y = load_data("x_small.txt","y_small.txt")
    X, y = load_data(args.x_file, args.y_file)
    vocab = get_vocab(X)
    print("Finishing loading data.")

    print("Encoding training data...")
    X_encoded = create_encoding(X, vocab)
    print("Finished encoding training data.")

    # Network parameters
    vocab_size = len(vocab) + 1
    input_size = len(X_encoded)
    hidden_size = args.hidden_size
    output_size = 10 # number of languages
    seq_len = len(X_encoded[0])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device == "cpu": # is using GPU, pin Dataloader to memory
        pin_memory = False
    else:
        pin_memory = True

    # Generate data
    print("Generating data...")
    X_gen,y_gen = gen_data(X_encoded,y)
    print("Number of sentences: ", len(X_gen))
    training_set  = Dataset(X_gen, y_gen)
    #training_generator = data.DataLoader(training_set, 200,pin_memory=pin_memory, shuffle=True)
    training_generator = data.DataLoader(training_set, args.batch_size, pin_memory=pin_memory, shuffle=True)
    print("Finishing processing training data.")

    print("Initilizing network model...")
    model = GRUNet(device,vocab_size,seq_len,input_size,hidden_size,output_size)
    model.init_model(device,vocab_size,seq_len,input_size,hidden_size,output_size)

    print("Training the network...")
    for i, (local_batch, local_labels) in enumerate(training_generator):
        print("Batch number {} of {}".format(i+1,len(X_gen)/args.batch_size))   
        #model.train(local_batch, local_labels, model, len(vocab), lr=0.1, epochs=20)
        model.train(local_batch, local_labels, model, len(vocab), args.learning_rate, args.epochs)

    # Save model to disk
    with open(args.model_name, 'wb+') as tmf:
        pickle.dump(model, tmf)
    # Save vocabulary integer mappings to disk
    with open("vocab", 'wb+') as f_voc:
        pickle.dump(vocab, f_voc)
    print("Vocabulary integer mappings saved to the file {}".format("vocab"))
    print("A trained model is saved to the file {}".format(args.model_name))

parser = argparse.ArgumentParser(description="Trains the model.")

parser.add_argument("-m", "--model_name", metavar="M", dest="model_name", type=str, help="The name of the network model.")
parser.add_argument("-x", "--x_file", metavar="X", dest="x_file", type=str,  help="File name of the language data.")
parser.add_argument("-y", "--y_file",metavar="Y", dest="y_file",type=str, help="File name of the language labels.")
parser.add_argument("-b", "--batch_size", metavar="B", dest="batch_size", type=int, default=100, help="Batch size used for for training the neural network (default 100).")
parser.add_argument("-e", "--epochs", metavar="E", dest="epochs", type=int,default=20, help="Number or epochs used for training the neural network (default 20).")
parser.add_argument("-r", "--learning_rate", metavar="R", dest="learning_rate", type=float, default=0.01, help="Optimizer learning rate (default 0.01).")
parser.add_argument("-l", "--hidden_size", metavar="L", dest="hidden_size", type=int, default=200, help="The size of the hidden layer (default 200).")

args = parser.parse_args()

# Sanity checks
if args.epochs < 0:
    exit("Error: Number of epochs can't be negative.")

if args.hidden_size < 0:
    exit("Error: Size of hidden layer can't be negative.")

if args.learning_rate < 0 or args.learning_rate > 1:
    exit("Error: Learning rate must be a float from 0 and lower than 1, e.g. 0.01.")

stop = time.time()
train_model()
start = time.time()
training_time = strftime("%H:%M:%S", gmtime(stop-start))
print("Time it took to train the model: ", training_time)

# python train_model.py -m trained_model -x x_small.txt -y y_small.txt -b 200 -e 20 -r 0.1 -l 300