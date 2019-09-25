import random
import numpy as np
from nltk.tokenize import WordPunctTokenizer

def load_data(x_datafile, y_datafile):
    Y = []
    X = []
    with open(y_datafile, encoding='utf-8') as labels:
        with open(x_datafile, encoding='utf-8') as data:
            for label_line, data_line in zip(labels, data):
                Y.append(label_line)
                X.append(data_line)
    return X, Y

X, Y = load_data("x.txt","y.txt")

def vectorize(X,Y):
    for label, data in zip(Y,X):
        data = list(data) # split on character labels
        print(data[:50])




vectorize(X,Y)