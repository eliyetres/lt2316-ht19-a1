import random
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(1)




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
