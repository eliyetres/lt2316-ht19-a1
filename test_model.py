import argparse
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import pandas as pd
import torch
from torch.utils import data

from dataloader import Dataset
from utils import (create_encoding, create_prefixes, gen_data, load_data,
                   load_pickle)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
if device == "cpu":  # if using GPU, pin Dataloader to memory
    pin_memory = False
else:
    pin_memory = True


def test_model():    
    print("Loading data...")
    #X, y, languages = load_data("x_test_small.txt", "y_test_small.txt")
    X, y, languages = load_data(args.x_file, args.y_file)
    # open the saved vocab file created in training
    vocab = load_pickle(args.vocab)
    inv_vocab = {v: k for k, v in vocab.items()}
    #vocab = load_pickle("vocab")
    print("Finishing loading data.")

    print("Encoding test data...")
    X_encoded = create_encoding(X, vocab)
    print("Finished encoding test data.")

    print("Loading model from {}.".format(args.model_name))
    trained_model = load_pickle(args.model_name)
    #trained_model = load_pickle("trained_model")  # load the model from disk
    trained_model.eval()
    print("Finished loading model.")
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
        print("Sentence number {} of {}.".format(batch_no+1, len(y)))
        batch_labels = []
        batch_sentences = []
        for sentence in batch:  # one sentence
            batch_sentences.append(sentence)
            batch_labels.append(b_label)
        test_set = Dataset(batch_sentences, batch_labels)
        test_generator = data.DataLoader(
            test_set, pin_memory=pin_memory, shuffle=False)
        all_labels[b_label]["total"] += 1

        for ii, (sentence, label) in enumerate(test_generator):
            #print("Prefix number {} of sentence {} of {} sentences.".format(ii+1,batch_no+1,len(y)))

            prefix = None
            sentence = sentence.to(device)
            label = label.to(device)

            char_len = torch.nonzero(sentence)  # measure number of chars
            char_len = char_len.size(0)
            #seq_len = len(sentence[0])
            seq_len = 1

            output = trained_model(sentence, seq_len)
            _, indices = torch.max(output.data, dim=1)
            lang = label.item()
            #print(torch.max(output.data, dim=1))

            int_ind = indices.item()
            all_labels[lang]["actual"].extend([lang])
            if indices[0] == label:  # correct prediction
                # prints full sentence without padding
                # including unknown characters, for debugging
                #decoded_sent = [inv_vocab[word_index] for word_index in sentence.tolist()[0] if word_index != 0]
                #decoded_sent = "".join(decoded_sent)
                #print(decoded_sent) # The encoded sentence
                print("Correcly classified sentence as {} at prefix #{}.".format(languages[int_ind], char_len))
                all_labels[lang]["correct"] += 1
                all_labels[lang]["predicted"].extend([int_ind])
                prefix = char_len
                break  # break the prediction loop if correct

            else:  # incorrect
                #print("Tried classifying {} as {} at prefix #{}.".format(languages[lang],languages[int_ind], char_len))
                all_labels[lang]["predicted"].extend([int_ind])

        all_labels[lang]["prefix"].append(prefix)


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
    print("Classification results:")
    print(df1_transposed)

    if args.predictions == True:
        print_predictions(all_labels,lang_no,languages)
  


def print_predictions(all_labels,lang_no,languages):
    """ Prints a table of how many times each language was predicted as the respective languages """
    all_predicts= []
    all_predicts_percent =[]
    for key in all_labels:        
        p = all_labels[key]["predicted"]
        #a = all_labels[key]["actual"]
        predicts = []
        percent_predicts = []
        for no in lang_no: # language number
            predicts.append(p.count(no))
        all_predicts.append(predicts)
        for i in predicts:
            summa = (i/sum(predicts))*100
            percent_predicts.append(summa)
        all_predicts_percent.append(percent_predicts)

    dft = pd.DataFrame(all_predicts)
    dft.index=[languages]
    dft.columns=[languages]

    dftp = pd.DataFrame(all_predicts_percent)
    dftp=dftp.round(2)
    dftp.index=[languages]
    dftp.columns=[languages]
    

    print("Predicted results per language:")
    print(dft)
    print("Predicted results in %:")
    print(dftp)

    

parser = argparse.ArgumentParser(description="Tests the model.")

parser.add_argument("-m", "--model_name", metavar="m", dest="model_name",
                    type=str, help="The previously saved network model.")
parser.add_argument("-x", "--x_file", metavar="X", dest="x_file",
                    type=str,  help="File name of the language data.")
parser.add_argument("-y", "--y_file", metavar="Y", dest="y_file",
                    type=str, help="File name of the language labels.")
parser.add_argument("-vo", "--vocab", metavar="VO", dest="vocab",
                    type=str, help="The previously saved vocabulary.")
parser.add_argument("-p", "--predictions", dest="predictions",
                    action='store_true',help="Prints a table with a sum and % of all languages' predictions.")

args = parser.parse_args()

test_model()

# python test_model.py -m model_1 -x processed_data/x_test.txt -y processed_data/y_test.txt -vo vocab -p