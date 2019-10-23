# LT2316 Machine Learning GU 2019
# Assignment #1: language identification with as few characters as possible

## Part 0: choosing "your" GPU

The GPU is cuda:0. 
The script will check if cuda is available, and then select `cuda:0`, if unavailable it will default to `cpu`.

## Part 1: data preparation

### Languages:

* Swedish
* Danish
* Norwegian (Bokmål)
* Icelandic
* Faroese
* English
* Welsh
* German
* Old English
* Arabic

I chose languages from similar language families, all the Scandinavian languages including Icelandic and Faroese, which could be described as a mix of Danish and Icelandic.  English, Old English and German are also similar and I wanted to see how well they can be predicted. Arabic was chosen because it's very much different from the other ones both alphabetically and syntactically. I wanted Arabic to be a sanity check for if the model is actually able to predict anything at all.

### Generating training and test files

The training and test files can be generated using the script create_data.py.

* `--showall`: Lists all the languages and their language codes
* `y_file`: path to the file with language labels for training
* `x_file`: path to the file with sentences for training
* `y_new`: name of new file with labels saved to disk
* `x_new`: name of new file with sentences saved to disk
* `languages`: Selected languages codes, separated by comma

The generated files in the for training and test files in the repo:

* x_train.txt
* y_train.txt
* x_test.txt
* y_test.txt

## Part 2: model and training

### Network model

The GRU network model is in the file `model.py`. The model has an input, embedding, hidden, linear and an output layer. It contains the critera for  CrossEntropyLoss, which combines LogSoftmax and NLLLoss in one single class, with and without reduction. The data is fed into the embedding layer which creates randomly-initialized vectors corresponding to the indices in the sentences. The hidden layer is initialized with the number of layers, length of the sentence and the layer's hidden size. The loss is calculated and printed for every trained batch.

The network model is trained by running the script `train_model.py` using the parameters:

* `--m`: The name of the network model.
* `--x_file`: File name of the language data.
* `--y_file` :File name of the language labels.
* `--vo`: File name of the vocabulary saved to disk.
* `--b` :Batch size used for for training the neural network (default 100).
* `--e`: Number or epochs used for training the neural network (default 20).
* `--r` :Optimizer learning rate (default 0.1).
* `--l`: The size of the hidden layer (default 200).
* `--t`: Loss type function to use (default 1) The loss functions are 1,2 and 3.
** Loss 1: Pytorch CrossEntropyLoss without reduction.
** Loss 2: CrossEntropyLoss multiplied with character prefix length (prefix lenth/sentence length).
** Loss 3: CrossEntropyLoss with additive with character prefix length.

### Training

The data is loaded from the selected files into training data, labels and vocabulary. The `utils.py` contains the functions for creating the  data fed into the network model.
First, the sentences are made into prefixes, made up of one list for every following character up to the lenth of the sentence.
The list of prefixes is encoded by swiching each character to the integer representation in the vocabulary, and are then padded up to the length of the longest sentence with zeroes.

The script will check if cuda is available, and then select `cuda:0`, if unavailable it will select `cpu`.

The encoded data is put into a Dataset from `dataset.py` and put into a dataloader. The dataloader pins the dataloader to memory if `cuda:0` is selected. Then it feeds the data in the selected number of batches into the model, printing the updated loss after each batch.

When the model has finished training, the model is saved to disk togeather with the vocabulary to be used in the evaluation.

## Part 3: evaluation

### Testing

The script will check if cuda is available, and then select `cuda:0`, if unavailable it will select `cpu`.
It loads the test data and labels and the vocabulary. 

For each sentence, the  prefixes and padding are done using the same functions as in training. For each sentence, an instance of n (length of sentence) prefixes are created and used for testing. The n prefixes are matched with the language label and fed into a dataloader (the reason for this was because it kept having the wrong shape otherwise) and it does not shuffle the data. Then, it tries to predict each prefix. If the prefix is correctly classified, the loop breaks and the testing continues with the next sentence. This turned out to make the testing considerably shorter, as some sentences were correctly predicted at the first character.

For every language, it saves all the predictions and the correct results. If a language was correctly predicted, the prefix number at which it was correct is also saved, otherwise the instance will be `None`. At a correct classification, the result is printed in the terminal.

The results for every language are saved as:
* Predicted
* Correct
* Percent of correctly predicted
* The mean prefix number at which is was correctly predicted
* The number of sentences for a language that were never predicted

## Part 4: reporting

### Results using different loss functions

The tabels show how many sentences that were correcly classified, at what percentage, the average prefix number at which is was correctly predicted and how many sentences that were never predicted correctly.

#### Loss 1

| Language           | Correct | % correct | Avg prefix | Never |
|--------------------|---------|-----------|------------|-------|
| Swedish            |         |           |            |       |
| Icelandic          |         |           |            |       |
| Danish             |         |           |            |       |
| Norwegian (Bokmål) |         |           |            |       |
| Faroese            |         |           |            |       |
| English            |         |           |            |       |
| Welsh              |         |           |            |       |
| German             |         |           |            |       |
| Old English        |         |           |            |       |
| Arabic             |         |           |            |       |
  
#### Loss 2

| Language           | Correct | % correct | Avg prefix | Never |
|--------------------|---------|-----------|------------|-------|
| Swedish            |         |           |            |       |
| Icelandic          |         |           |            |       |
| Danish             |         |           |            |       |
| Norwegian (Bokmål) |         |           |            |       |
| Faroese            |         |           |            |       |
| English            |         |           |            |       |
| Welsh              |         |           |            |       |
| German             |         |           |            |       |
| Old English        |         |           |            |       |
| Arabic             |         |           |            |       |

#### Loss 3

| Language           | Correct | % correct | Avg prefix | Never |
|--------------------|---------|-----------|------------|-------|
| Swedish            |         |           |            |       |
| Icelandic          |         |           |            |       |
| Danish             |         |           |            |       |
| Norwegian (Bokmål) |         |           |            |       |
| Faroese            |         |           |            |       |
| English            |         |           |            |       |
| Welsh              |         |           |            |       |
| German             |         |           |            |       |
| Old English        |         |           |            |       |
| Arabic             |         |           |            |       |

## Part Bonus A: mini-batching 

## Part Bonus B: GRUCell (7 points)