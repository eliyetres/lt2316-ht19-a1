import random
import numpy as np
from nltk.tokenize import WordPunctTokenizer

def load_data(y_datafile,x_datafile):
    Y = []
    X = []
    with open(y_datafile, encoding='utf-8') as labels:
        with open(x_datafile, encoding='utf-8') as data:
            for label_line, data_line in zip(labels, data):
                Y.append(label_line)
                X.append(data_line)
    return Y, X
