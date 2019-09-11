from read import read_labels, open_datafiles
# Ignore everything past the first 100 characters of each sentence and remove any instances that have less than 100 characters. 


# Requirements: 
# Implement the learning algorithm in PyTorch.

# Use a recurrent neural network variant, such as a Gated Recurrent Unit (GRU) network, be central to the classifiers design. (You will also probably need an output softmax activation layer.)


# Choose any 10 languages and document them in your readme. Split out those languages into smaller training and test files and submit them with your project.

dirr = dirr = "data/" # Data directory

languages = ["Swedish", "Danish", "Bokmål", "Icelandic", "Faroese", "English", "Welsh", "Cornish", "Breton", "Old English "] # Selected languages

lan = read_labels(dirr, "labels.csv", languages) # Language codes

x_train, y_train = open_datafiles(dirr, "y_train.txt", "x_train.txt",lan)

x_test, y_test = open_datafiles(dirr, "y_test.txt", "x_test.txt",lan)