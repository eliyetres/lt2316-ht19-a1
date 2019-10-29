# Assignment #1: language identification with as few characters as possible

## Part 0: choosing "your" GPU

The GPU is cuda:0.
The script will check if cuda is available, and then select `cuda:0`, if unavailable it will default to `cpu`.

## Part 1: data preparation

### Languages

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

* `--showall`: Lists all the languages and their language codes.
* `y_file`: path to the file with language labels for training.
* `x_file`: path to the file with sentences for training.
* `y_new`: name of new file with labels saved to disk.
* `x_new`: name of new file with sentences saved to disk.
* `languages`: Selected languages codes, separated by comma.

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
* `--y_file`: File name of the language labels.
* `--vo`: File name of the vocabulary saved to disk.
* `--b`: Batch size used for for training the neural network (default 100).
* `--e`: Number or epochs used for training the neural network (default 20).
* `--r`: Optimizer learning rate (default 0.1).
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

* Loss: 0.0007443547365255654

| Language           | Correct | Total | % correct | Avg prefix | Never |
|--------------------|---------|-------|-----------|------------|-------|
| Old English        | 474     | 500   | 94        |    6       |  26   |
| Arabic             | 499     | 500   | 99        |    1       |   1   |
| Welsh              | 488     | 500   | 97        |    2       |  12   |
| Danish             | 444     | 500   | 88        |    6       |  56   |
| German             | 477     | 500   | 95        |    4       |  23   |
| English            | 477     | 500   | 95        |    6       |  23   |
| Faroese            | 482     | 500   | 96        |    3       |  18   |
| Icelandic          | 463     | 500   | 92        |    6       |  37   |
| Norwegian (Bokmål) | 431     | 500   | 86        |   10       |  69   |
| Swedish            | 462     | 500   | 92        |    8       |  38   |

### Total number of predicted results per language in %

Read from the left column: the model tried to predict arabic as arabic 77.36% of the times. These percentages do not include number of unsucessful predictions.

|     | ang   | ara   | cym   | dan   | deu   | eng   | fao   | isl   | nob   | swe   |
|-----|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| ang | 8.29  | 0.00  | 13.67 | 7.62  | 14.39 | 23.20 | 10.24 | 10.05 | 5.63  | 6.91  |
| ara | 0.47  | 77.36 | 0.93  | 8.37  | 1.71  | 6.67  | 3.88  | 0.16  | 0.00  | 0.47  |
| cym | 4.96  | 0.00  | 19.83 | 12.15 | 13.90 | 23.57 | 15.08 | 1.42  | 5.24  | 3.86  |
| dan | 4.16  | 0.00  | 7.18  | 5.05  | 17.90 | 8.11  | 10.64 | 3.54  | 34.93 | 8.50  |
| deu | 7.37  | 0.00  | 9.39  | 12.01 | 10.96 | 11.90 | 20.53 | 12.72 | 7.92  | 7.19  |
| eng | 13.52 | 0.00  | 15.28 | 11.82 | 12.70 | 8.43  | 11.41 | 7.81  | 13.18 | 5.85  |
| fao | 5.81  | 0.00  | 9.34  | 23.84 | 4.66  | 10.58 | 13.60 | 16.51 | 8.89  | 6.77  |
| isl | 6.68  | 0.00  | 6.00  | 5.99  | 5.49  | 10.16 | 47.90 | 6.60  | 4.51  | 6.68  |
| nob | 4.66  | 0.00  | 5.12  | 42.14 | 8.66  | 6.96  | 12.12 | 6.46  | 3.49  | 10.38 |
| swe | 7.27  | 0.00  | 13.25 | 20.68 | 8.55  | 8.11  | 13.96 | 6.29  | 16.10 | 5.78  |

### Total number of predicted results per language

|     | ang | ara | cym  | dan  | deu  | eng  | fao  | isl | nob  | swe  |
|-----|-----|-----|------|------|------|------|------|-----|------|------|
| ang | 474 | 0   | 782  | 436  | 823  | 1327 | 586  | 575 | 322  | 395  |
| ara | 3   | 499 | 6    | 54   | 11   | 43   | 25   | 1   | 0    | 3    |
| cym | 122 | 0   | 488  | 299  | 342  | 580  | 371  | 35  | 129  | 95   |
| dan | 366 | 0   | 631  | 444  | 1573 | 713  | 935  | 311 | 3070 | 747  |
| deu | 321 | 0   | 409  | 523  | 477  | 518  | 894  | 554 | 345  | 313  |
| eng | 765 | 0   | 865  | 669  | 719  | 477  | 646  | 442 | 746  | 331  |
| fao | 206 | 0   | 331  | 845  | 165  | 375  | 482  | 585 | 315  | 240  |
| isl | 468 | 0   | 421  | 420  | 385  | 712  | 3358 | 463 | 316  | 468  |
| nob | 575 | 0   | 631  | 5197 | 1068 | 858  | 1495 | 797 | 431  | 1280 |
| swe | 581 | 0   | 1058 | 1652 | 683  | 648  | 1115 | 502 | 1286 | 462  |

#### Loss 2

#### Loss 3

## Part Bonus A: mini-batching

There is an option of feeding the data into the model using batches. One batch is one prefix of a sentence. Since the data is already generated and put into a dataloader, the data is shuffeled before it's fed into the model.

## Part Bonus B: GRUCell (7 points)
