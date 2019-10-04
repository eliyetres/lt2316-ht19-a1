import argparse
import time

from get_data import (load_data, get_vocab, split_data, create_encoding)

start_time = time.time()
batch_size = 50
X_raw, y_raw = load_data("x.txt","y.txt")
vocab = get_vocab(X_raw)
print("Finishing loading data after ", time.time() - start_time)
start_time = time.time()
print("Processing data...")
X, y = create_encoding(X_raw, y_raw, vocab)
print("Finishing processing data after ", time.time() - start_time)
print("Splitting data...")
X_train, X_test, y_train, y_test = split_data(X,y)
print("Finished splitting data.")

print(type(X_train[0]), type(y_train[0]))

parser = argparse.ArgumentParser(description="Trains the model.")

parser.add_argument("-m", "--model", metavar="m", dest="model", type=str, help="The network model.")

args = parser.parse_args()
