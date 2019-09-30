import re 
import random
#import numpy as np
import torch
from nltk.tokenize import WordPunctTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np

torch.manual_seed(1)

def load_data(x_datafile, y_datafile):
    """Loads the data and labels from the files when selecting languages.
    
    Arguments:
        x_datafile {file} -- File containing data
        y_datafile {file} -- File containing labels
    
    Returns:
        X:      list -- Data
        Y:      list -- Labels
        vocab:  dict -- The character vocabulary
    """
    Y = []
    X = []
    with open(y_datafile, "r", encoding='utf-8') as labels:
        with open(x_datafile, "r", encoding='utf-8') as data:
            #labels = labels.read()
            data_set = re.sub(r'(\n)', '', data.read())                
            vocab = {f:i for i,f in enumerate(sorted(list(set(data_set))))}
            for label_line, data_line in zip(labels, data):
                print(data_line)
                print(label_line)
                Y.append(label_line)
                X.append(data_line)
        
    return X, Y, vocab

X, Y, vocab = load_data("x.txt","y.txt")


text ="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore."


voc = dict(enumerate(tuple(set(text))))
#print(voc)

def sent_padding(sentence):
    """ Pads a sentence with zeros times the length of the sentece.
    
    Arguments:
        sentence {string} -- One sentence
    
    Returns:
        dict -- The percentage of each sentence completed and the padded sentence.
    """
    #padded_sents = []
    prefix_percent = {}
    ch = ""
    len_sent = len(sentence)    
    for c, i in zip(sentence, range(len_sent)):        
        ch+=c
        p_complete = (i+1)/len(sentence)
        #print(p_complete)
        new_sent = str(ch).ljust(len_sent, '0')       
        #padded_sents.append(new_sent)
        prefix_percent[p_complete] = new_sent

    pretty = json.dumps(prefix_percent, indent=4)
    print(pretty)
    return prefix_percent


#sent_padding(text)


def encodings(vocab, sentence):
    """"Encoding by mapping each character in a sentence to a correspoding index in a vocabulary.

    Arguments:
        vocab {dict} -- [description]
        sentence {string} -- [description]
    
    Returns:
        numpy array -- The encoded sentence. 
    """
    char2int = {ch: ii for ii, ch in vocab.items()}
    #encoded = np.array([char2int[ch] for ch in text])
    #encoded = [char2int[ch] for ch in text]
    encoded_tensor = torch.LongTensor([char2int[ch] for ch in text])
    #print(encoded)
    print(encoded_tensor)
    return encoded_tensor

#encodings(voc, text)




def one_hot_embedding(vocab, character):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    num_classes = len(vocab)
    dim = len(vocab) + 1
    #word_to_ix = {"hello": 0, "world": 1}
    embeds = nn.Embedding(num_classes, dim)  # words in vocab, dimensional embeddings
    lookup_tensor = torch.tensor([vocab[character]], dtype=torch.long)
    embed = embeds(lookup_tensor)
    print(embed)
    return embed



#one_hot_embedding(vocab, "Ö")