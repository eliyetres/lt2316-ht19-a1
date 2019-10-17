import random
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

torch.manual_seed(1)

def load_pickle(filename):
    pickle_load = pickle.load(open(filename, 'rb'))
    return pickle_load

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


def cross_entropy_cat(X,y, epsil=1e-12):  
    """
    Function for calculating cross entropy, probably very bad
    """          
    m = y.shape[0]
    p = torch.softmax(X+epsil, dim=1)
    # Extracting softmax probability of the correct label for each sample
    log_likelihood = -torch.log(p[range(m),y]+epsil)    # added epsil to avoid log(0)
    loss = torch.sum(log_likelihood)/m

    return loss


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def my_cross_entropy(output, label, prefix):
    x_terms = -torch.gather(output, 1, label.unsqueeze(1)).squeeze()
    log_terms = torch.log(torch.sum(torch.exp(output), dim=1))
    prefixes = torch.gather(prefix, 0, label).float()
    return torch.mean((x_terms+log_terms)*prefixes/len(output))
