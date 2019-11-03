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
** Loss 1: CrossEntropyLoss with mean reduction.
** Loss 2: CrossEntropyLoss without reduction multiplied with character prefix length (prefix lenth/sentence length).
** Loss 3: CrossEntropyLoss without reduction with additive character prefix length.

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

* The predicted language
* The language that was correct
* Percentages of the correctly predicted languages
* The mean prefix number at which each language was correctly predicted
* The number of sentences for a language that were never predicted

## Part 4: reporting

### Results using different loss functions

The tabels show how many sentences that were correcly classified, at what percentage, the average prefix number at which is was correctly predicted and how many sentences that were never predicted correctly.

The extra tabels show the sum and percentages of all the predictions made for each language. One prediction equals one prefix in a sentence, so sentences that were harder and took longer to classify will have a higher sum. Since the first table only shows how many full sentences were correct this shows more in detail what the model has tried to predict.

|                | Loss1   | Loss2   | Loss3    |
|----------------|---------|---------|----------|
| Total accuracy | 88.8%   | 90.0%   | 89.1%    |
| Avg loss       | 0.0007  | 72.855  | 31542.24 |
| Avg prefix     | 4.1     | 4.5     | 4.1      |

The models performs very similar. The loss for model 1 is the lowest and predicts the correct language at an earlier prefix than the second one, but model 2 (multiplying the prefix lengths) gave the most accurate predictions overall.

#### Loss 1

| Language           | Correct | Total | % correct | Avg prefix | Never |
|--------------------|---------|-------|-----------|------------|-------|
| Old English        | 459     | 500   | 91        |   4        |  41   |
| Arabic             | 499     | 500   | 99        |   1        |   1   |
| Welsh              | 480     | 500   | 96        |   3        |  20   |
| Danish             | 414     | 500   | 82        |   3        |  86   |
| German             | 443     | 500   | 88        |   6        |  57   |
| English            | 430     | 500   | 86        |   3        |  70   |
| Faroese            | 465     | 500   | 93        |   3        |  35   |
| Icelandic          | 408     | 500   | 81        |   9        |  92   |
| Norwegian (Bokmål) | 414     | 500   | 82        |   4        |  86   |
| Swedish            | 432     | 500   | 86        |   5        |  68   |

##### Loss 1: Total number of predicted results per language in %

|     | ang   | ara   | cym   | dan   | deu   | eng   | fao   | isl   | nob   | swe   |
|-----|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| ang | 7.50  | 0.20  | 6.10  | 22.55 | 13.13 | 22.89 | 11.30 | 7.98  | 3.47  | 4.89  |
| ara | 0.48  | 80.48 | 0.00  | 0.00  | 12.10 | 0.16  | 3.06  | 2.42  | 1.29  | 0.00  |
| cym | 7.35  | 0.03  | 13.21 | 10.87 | 13.84 | 19.59 | 11.92 | 3.05  | 8.31  | 11.83 |
| dan | 6.08  | 0.01  | 1.56  | 3.93  | 6.98  | 10.89 | 12.71 | 2.30  | 37.81 | 17.75 |
| deu | 9.01  | 0.05  | 4.81  | 18.92 | 5.01  | 12.46 | 13.49 | 6.08  | 17.35 | 12.84 |
| eng | 15.38 | 0.01  | 10.17 | 13.40 | 9.81  | 4.81  | 7.40  | 4.05  | 12.26 | 22.72 |
| fao | 8.41  | 0.04  | 3.02  | 19.48 | 1.66  | 9.43  | 9.18  | 30.85 | 13.52 | 4.42  |
| isl | 7.50  | 0.18  | 2.21  | 6.72  | 4.23  | 3.48  | 61.30 | 2.89  | 6.83  | 4.65  |
| nob | 3.87  | 0.00  | 1.52  | 52.25 | 1.90  | 7.85  | 14.60 | 1.18  | 3.90  | 12.93 |
| swe | 3.89  | 0.52  | 4.99  | 29.60 | 8.38  | 6.95  | 10.66 | 7.82  | 22.75 | 4.44  |

##### Loss 1: Total number of predicted results per language

|     | ang  | ara | cym | dan  | deu | eng  | fao  | isl  | nob  | swe  |
|-----|------|-----|-----|------|-----|------|------|------|------|------|
| ang | 459  | 12  | 373 | 1379 | 803 | 1400 | 691  | 488  | 212  | 299  |
| ara | 3    | 499 | 0   | 0    | 75  | 1    | 19   | 15   | 8    | 0    |
| cym | 267  | 1   | 480 | 395  | 503 | 712  | 433  | 111  | 302  | 430  |
| dan | 641  | 1   | 164 | 414  | 736 | 1148 | 1340 | 242  | 3986 | 1871 |
| deu | 796  | 4   | 425 | 1672 | 443 | 1101 | 1192 | 537  | 1534 | 1135 |
| eng | 1375 | 1   | 909 | 1198 | 877 | 430  | 662  | 362  | 1096 | 2032 |
| fao | 426  | 2   | 153 | 987  | 84  | 478  | 465  | 1563 | 685  | 224  |
| isl | 1058 | 26  | 311 | 947  | 597 | 491  | 8642 | 408  | 963  | 656  |
| nob | 411  | 0   | 161 | 5550 | 202 | 834  | 1551 | 125  | 414  | 1373 |
| swe | 378  | 51  | 485 | 2879 | 815 | 676  | 1037 | 761  | 2213 | 432  |

#### Loss 2

Avgerage percent correct for model: 89.5%

| Language           | Correct | Total | % correct | Avg prefix | Never |
|--------------------|---------|-------|-----------|------------|-------|
| Old English        | 454     | 500   | 90        |   5        |  46   |
| Arabic             | 499     | 500   | 99        |   1        |   1   |
| Welsh              | 486     | 500   | 97        |   2        |  14   |
| Danish             | 394     | 500   | 78        |   6        | 106   |
| German             | 456     | 500   | 91        |   6        |  44   |
| English            | 447     | 500   | 89        |   7        |  53   |
| Faroese            | 474     | 500   | 94        |   3        |  26   |
| Icelandic          | 409     | 500   | 81        |   7        |  91   |
| Norwegian (Bokmål) | 451     | 500   | 90        |   3        |  49   |
| Swedish            | 432     | 500   | 86        |   5        |  68   |

##### Loss 2: Total number of predicted results per language in %

|     | ang   | ara   | cym   | dan   | deu   | eng   | fao   | isl   | nob   | swe   |
|-----|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| ang | 6.23  | 0.00  | 5.67  | 9.48  | 10.68 | 42.37 | 7.95  | 5.67  | 7.32  | 4.64  |
| ara | 1.60  | 79.97 | 0.16  | 0.16  | 0.64  | 0.32  | 1.44  | 2.56  | 0.80  | 12.34 |
| cym | 6.63  | 0.28  | 19.30 | 8.26  | 10.13 | 15.29 | 14.06 | 8.26  | 10.96 | 6.83  |
| dan | 5.14  | 0.00  | 2.37  | 2.86  | 10.31 | 8.39  | 12.04 | 3.67  | 42.74 | 12.49 |
| deu | 11.13 | 0.04  | 8.99  | 9.70  | 5.97  | 4.99  | 16.40 | 9.33  | 21.90 | 11.56 |
| eng | 20.27 | 0.01  | 6.38  | 8.44  | 8.79  | 5.06  | 12.36 | 5.19  | 15.03 | 18.48 |
| fao | 4.06  | 0.04  | 6.12  | 17.84 | 2.40  | 15.10 | 10.62 | 25.81 | 11.38 | 6.63  |
| isl | 6.05  | 0.22  | 2.63  | 4.73  | 3.15  | 2.50  | 67.16 | 3.23  | 6.00  | 4.32  |
| nob | 4.73  | 0.14  | 5.20  | 28.11 | 7.64  | 12.57 | 18.56 | 3.81  | 6.80  | 12.44 |
| swe | 3.77  | 0.00  | 4.52  | 14.97 | 8.80  | 8.63  | 18.81 | 5.48  | 30.43 | 4.58  |

##### Loss 2: Total number of predicted results per language

|     | ang  | ara | cym | dan  | deu  | eng  | fao  | isl  | nob  | swe  |
|-----|------|-----|-----|------|------|------|------|------|------|------|
| ang | 454  | 0   | 413 | 691  | 778  | 3087 | 579  | 413  | 533  | 338  |
| ara | 10   | 499 | 1   | 1    | 4    | 2    | 9    | 16   | 5    | 77   |
| cym | 167  | 7   | 486 | 208  | 255  | 385  | 354  | 208  | 276  | 172  |
| dan | 708  | 0   | 327 | 394  | 1421 | 1156 | 1660 | 506  | 5891 | 1721 |
| deu | 850  | 3   | 687 | 741  | 456  | 381  | 1253 | 713  | 1673 | 883  |
| eng | 1792 | 1   | 564 | 746  | 777  | 447  | 1093 | 459  | 1329 | 1634 |
| fao | 181  | 2   | 273 | 796  | 107  | 674  | 474  | 1152 | 508  | 296  |
| isl | 765  | 28  | 333 | 598  | 399  | 316  | 8497 | 409  | 759  | 547  |
| nob | 314  | 9   | 345 | 1865 | 507  | 834  | 1231 | 253  | 451  | 825  |
| swe | 356  | 0   | 426 | 1412 | 830  | 814  | 1774 | 517  | 2870 | 432  |

#### Loss 3

| Language           | Correct | Total | % correct | Avg prefix | Never |
|--------------------|---------|-------|-----------|------------|-------|
| Old English        | 460     | 500   | 92        |   5        |  40   |
| Arabic             | 499     | 500   | 99        |   1        |   1   |
| Welsh              | 484     | 500   | 96        |   2        |  16   |
| Danish             | 345     | 500   | 69        |   4        | 155   |
| German             | 460     | 500   | 92        |   5        |  40   |
| English            | 449     | 500   | 89        |   6        |  51   |
| Faroese            | 447     | 500   | 89        |   4        |  53   |
| Icelandic          | 423     | 500   | 84        |   5        |  77   |
| Norwegian (Bokmål) | 455     | 500   | 91        |   3        |  45   |
| Swedish            | 430     | 500   | 86        |   6        |  70   |

##### Loss 3: Total number of predicted results per language in %

|     | ang   | ara   | cym   | dan   | deu   | eng   | fao   | isl   | nob   | swe   |
|-----|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| ang | 6.92  | 0.00  | 6.53  | 7.84  | 11.45 | 38.58 | 7.25  | 9.36  | 8.14  | 3.93  |
| ara | 0.80  | 79.97 | 0.32  | 0.00  | 1.60  | 0.32  | 0.80  | 0.48  | 0.32  | 15.38 |
| cym | 7.30  | 0.03  | 16.91 | 6.88  | 16.28 | 22.91 | 7.37  | 3.07  | 10.79 | 8.45  |
| dan | 4.18  | 0.00  | 1.36  | 1.93  | 9.44  | 9.31  | 9.84  | 3.67  | 51.83 | 8.44  |
| deu | 11.61 | 0.47  | 3.01  | 10.51 | 6.62  | 12.95 | 14.77 | 13.91 | 14.34 | 11.81 |
| eng | 22.58 | 0.12  | 8.72  | 10.33 | 8.16  | 5.39  | 6.56  | 8.66  | 17.79 | 11.68 |
| fao | 6.99  | 0.04  | 4.69  | 14.15 | 5.08  | 8.39  | 6.08  | 33.51 | 14.65 | 6.43  |
| isl | 5.49  | 0.01  | 1.52  | 2.99  | 5.03  | 4.36  | 61.83 | 4.12  | 11.06 | 3.58  |
| nob | 5.56  | 0.00  | 2.57  | 35.87 | 8.75  | 10.88 | 10.72 | 7.48  | 7.16  | 11.02 |
| swe | 6.16  | 0.31  | 4.63  | 17.13 | 10.49 | 9.02  | 10.44 | 12.35 | 25.33 | 4.13  |

##### Loss 3: Total number of predicted results per language

|     | ang  | ara | cym | dan  | deu  | eng  | fao  | isl  | nob  | swe  |
|-----|------|-----|-----|------|------|------|------|------|------|------|
| ang | 460  | 0   | 434 | 521  | 761  | 2564 | 482  | 622  | 541  | 261  |
| ara | 5    | 499 | 2   | 0    | 10   | 2    | 5    | 3    | 2    | 96   |
| cym | 209  | 1   | 484 | 197  | 466  | 656  | 211  | 88   | 309  | 242  |
| dan | 748  | 0   | 244 | 345  | 1690 | 1666 | 1762 | 656  | 9277 | 1510 |
| deu | 807  | 33  | 209 | 731  | 460  | 900  | 1027 | 967  | 997  | 821  |
| eng | 1879 | 10  | 726 | 860  | 679  | 449  | 546  | 721  | 1481 | 972  |
| fao | 514  | 3   | 345 | 1041 | 374  | 617  | 447  | 2465 | 1078 | 473  |
| isl | 564  | 1   | 156 | 307  | 517  | 448  | 6351 | 423  | 1136 | 368  |
| nob | 353  | 0   | 163 | 2279 | 556  | 691  | 681  | 475  | 455  | 700  |
| swe | 641  | 32  | 482 | 1783 | 1092 | 939  | 1087 | 1286 | 2637 | 430  |

## Part Bonus A: mini-batching

I've added the option of feeding the data into the model using batches. One batch is one prefix of a sentence.
I did a test on a smaller dataset of only 3 languages and plotted the results over 20 epochs with batches of different sizes. The loss is lower when selecting a higher batch size and the loss quickly goes down and doesn't change much.

![80 batches](../results/plot80.png)
![60 batches](../results/plot60.png)
![40 batches](../results/plot40.png)
![20 batches](../results/plot20.png)
