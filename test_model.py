import argparse
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils import data

from dataloader import Dataset
from get_data import create_encoding, create_prefixes, gen_data, load_data
from utils import load_pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device == "cpu":  # if using GPU, pin Dataloader to memory
    pin_memory = False
else:
    pin_memory = True

print("Loading data...")
X, y, languages = load_data("x_small_test.txt", "y_small_test.txt")
#X, y = load_data(args.x_file, args.y_file)
vocab = load_pickle("vocab")  # open the saved vocab file created in training
print("Finishing loading data.")

print("Encoding test data...")
X_encoded = create_encoding(X, vocab)
print("Finished encoding test data.")

#print("Loading model from {}.".format(args.model_name))
print("Loading model...")
#trained_model = load_model(args.model_name)
trained_model = load_pickle("trained_model")  # load the model from disk
print("Finished loading model.")

# Generate data
print("Generating data...")
print("Finishing processing test data.")


languages = list(languages.keys())
# Output variables
lang_no = list(sorted(set(y)))  # languages
all_labels = {el: {
    "correct": 0,
    "total": 0,
    "prefix": [],
    "predicted": [],
    "actual": []
} for el in lang_no}  # create indices for labels


# create 100 sentences out of one
for batch_no, (batch, b_label) in enumerate(zip(X_encoded, y)):  # tensor 100x100
    print("Batch number {} of {}.".format(batch_no+1, len(y)))
    batch_labels = []
    batch_sentences = []
    for sentence in batch:  # one sentence
        batch_sentences.append(sentence)
        batch_labels.append(b_label)

    test_set = Dataset(batch_sentences, batch_labels)
    test_generator = data.DataLoader(
        test_set, pin_memory=pin_memory, shuffle=False)
    all_labels[b_label]["total"] += 1
    predicted = []
    actual = []

    for ii, (sentence, label) in enumerate(test_generator):
        #print("Prefix number {} of batch {} of {} batches.".format(ii+1,batch_no+1,len(y)))

        prefix = None
        sentence = sentence.to(device)
        label = label.to(device)

        char_len = torch.nonzero(sentence)  # measure number of chars
        char_len = char_len.size(0)
        seq_len = len(sentence[0])

        output = trained_model(sentence, seq_len)
        _, indices = torch.max(output.data, dim=1)
        lang = label.item()
        #print(torch.max(output.data, dim=1))

        all_labels[lang]["actual"].append(label)
        int_ind = indices.item()

        actual.append(lang)
        if indices[0] == label:  # correct prediction
            print("Correcly classified sentence as {} at prefix #{}.".format(
                languages[int_ind], char_len))
            all_labels[lang]["correct"] += 1
            predicted.append(int_ind)
            prefix = char_len
            break  # break the prediction loop if correct

        else:  # incorrect
            predicted.append(int_ind)

    all_labels[lang]["prefix"].append(prefix)
    all_labels[lang]["predicted"].extend(predicted)
    all_labels[lang]["actual"].extend(actual)


results = {el: {
    "correct": 0,
    "total": 0,
    "percent": 0,
    "avg_prefix": [],
    "never_predicted": [],
} for el in lang_no}  # create indices for labels


for key in all_labels:
    correct = all_labels[key]["correct"]
    total = all_labels[key]["total"]
    percent = (correct/total)*100
    prefix = all_labels[key]["prefix"]
    avg_prefix = sum(filter(None, prefix))/len(prefix)
    never_predicted = prefix.count(None)

    results[key]["correct"] = correct
    results[key]["total"] = total
    results[key]["percent"] = percent
    results[key]["avg_prefix"] = avg_prefix
    results[key]["never_predicted"] = never_predicted


df = pd.DataFrame(results).astype(int)
df.round(0)
df.columns = list(languages)
df.index = ["Correct", "Total", "%", "Avg prefix len", "Never predicted"]

df1_transposed = df.transpose()


print(df)
print(df1_transposed)
# parser = argparse.ArgumentParser(description="Tests the model.")

# parser.add_argument("-m", "--model_name", metavar="m", dest="model_name", type=str, help="The previously saved network model.")

# args = parser.parse_args()
