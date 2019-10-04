import random
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from nltk.tokenize import WordPunctTokenizer
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(1)

def create_prefixes(sentence):
    """ 
    Creates a hundred instances representing prefixes of one sentence. Returns a list of the prefixes. 
    """
    padded_sents = []
    sent_str = ""
    for s in (sentence):
        sent_str += s
        padded_sents.append(sent_str)

    return padded_sents


def encodings(vocab, sentence):
    """"
    Encoding by mapping each character in a sentence to a correspoding index in a vocabulary.
    """
    encoded = [vocab[ch] for ch in sentence]
    encoded_tensor = torch.LongTensor(encoded)

    return encoded_tensor


def padding(encoded_tensors):
    """ 
    Pads a list of tensors with zeros.
    """
    #print("Length of tensors: ", len(encoded_tensors))
    padded_tensors = pad_sequence(encoded_tensors, batch_first=True, padding_value=0)

    return padded_tensors


def yield_batches(iterable, batch_size):
    """ Splits the data into batches using yield
    
    Arguments:
        iterable {function} -- range(start,end)
        batch_size {int} -- The size of each batch

        Example: for x in batches(range(0, 10), 100):
    """
    endpoint = len(iterable)
    for ndx in range(0, endpoint, batch_size):

        yield iterable[ndx:min(ndx + batch_size, endpoint)]


def cross_entropy_cat(self, X,y, epsil=1e-12):  
    """
    Function for calculating cross entropy, probably very bad
    """          
    m = y.shape[0]
    p = torch.softmax(X+epsil, dim=1)                                                                                                                                   
    # Extracting softmax probability of the correct label for each sample
    log_likelihood = -torch.log(p[range(m),y]+epsil)    # added epsil to avoid log(0)
    loss = torch.sum(log_likelihood)/m

    return loss
