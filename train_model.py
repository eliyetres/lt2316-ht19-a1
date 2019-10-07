import argparse
import time
import torch
from get_data import create_encoding, load_data, get_vocab, split_data
from model import GRUNet

############
batch_size = 20
############


print("Loading data...")
X , y = load_data("x.txt","y.txt")
vocab = get_vocab(X)
print("Finishing loading data.")

print(type(X[0]), type(y[0]))

print("Splitting data...")
X_train, X_test, y_train, y_test = split_data(X,y)
print("Finished splitting data.")

print("Encoding training data...")
X_train, y_train = create_encoding(X_train, y_train, vocab)
print("Finishing processing training data.")

print("Training the network...")
vocab_size = len(vocab) + 1
input_size = len(X_train)
hidden_size = 300
output_size = 10 # number of languages
batch_len = len(X_train[0])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = GRUNet(device,vocab_size, batch_len,  input_size=input_size,  
                       hidden_size=hidden_size, output_size=output_size)

model.init_model(device, vocab_size, batch_len, input_size, hidden_size, output_size)




parser = argparse.ArgumentParser(description="Trains the model.")

parser.add_argument("-m", "--model", metavar="m", dest="model", type=str, help="The network model.")
parser.add_argument("-x", "--x_file", type=str,  help="File name of the language data.")
parser.add_argument("-y", "--y_file", type=str, help="File name of the language labels.")
parser.add_argument("-b", "--batch_size", metavar="B", dest="batch", type=int,default=100, help="Batch size used for for training the neural network (default 100).")
parser.add_argument("-e", "--epochs", metavar="E", dest="epochs", type=int,default=20, help="Number or epochs used for training the neural network (default 20).")
parser.add_argument("-r", "--learning_rate", metavar="R", dest="learning_rate", type=float, default=0.01, help="Optimizer learning rate (default 0.01).")
parser.add_argument("-h", "--hidden_size", metavar="H", dest="hidden_size", type=float, default=0.01, help="The size of the hidden layer (default ).")


args = parser.parse_args()


# Sanity checks
if args.epochs < 0:
    exit("Error: Number of epochs can't be negative")

if args.learning_rate < 0 or args.learning_rate > 1:
    exit("Error: Learning rate must be a float from 0 and lower than 1, e.g. 0.01")

if args.batch >= len(args.x_file):
    exit("Error: training batch must be lower than total training data.")
