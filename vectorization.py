import re 
import random
import numpy as np
import torch
from nltk.tokenize import WordPunctTokenizer

def load_data(x_datafile, y_datafile):
    Y = []
    X = []
    with open(y_datafile, encoding='utf-8') as labels:
        with open(x_datafile, encoding='utf-8') as data:
            data_set = re.sub(r'(\n)', '', data.read())            
            vocab = {f:i for i,f in enumerate(sorted(list(set(data_set))))}
            for label_line, data_line in zip(labels, data):
                Y.append(label_line)
                X.append(data_line)
    return X, Y, vocab

X, Y, vocab = load_data("x.txt","y.txt")

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """

    num_embeddings =len(vocab)+1


    e = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None)


    print(num_classes)
    
    return None

    def make_tensor(X, Y):
        if device == "cpu":
            # Using CPU  
            X = torch.Tensor(X)
            Y = torch.Tensor(Y)
        else:
            # Using GPU                             
            X = torch.as_tensor(X, dtype=torch.float, device=self.device)
            Y = torch.as_tensor(Y, dtype=torch.float, device=self.device)
        return X, Y


one_hot_embedding(vocab, len(vocab))