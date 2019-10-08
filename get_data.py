import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


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
    labels =  open(y_datafile, "r", encoding='utf-8').read().split("\n")
    data = open(x_datafile, "r", encoding='utf-8').read().split("\n")
    languages = {f:i for i,f in enumerate(sorted(list(set(labels))))} # create indices for labels
    for label_line, data_line in zip(labels, data):
        Y.append(languages[label_line])
        X.append(data_line)
        
    return X, Y

def get_vocab(data):
    """ Gets the vocabulary for the sentences """
    sents = [[x for x in sent] for sent in data]
    vocab = {f:i for i,f in enumerate(sorted(list(set(sum(sents, [])))))}

    return vocab

def create_encoding(X, vocab):
    """ Creates padded prefixes for the sentences and converts the int labels to tensors. """
    data = []
    # labels = []
    # print(y)
    # for l in y:
    #     tensor_label = torch.LongTensor(l)
    #     labels.append(tensor_label)
    #     print(tensor_label)

    for sentence in X:
        if len(sentence) == 0: # In case file is saved with ending newline
            return
        sentence_prefixes = create_prefixes(sentence)
        sents_encodings = []
        for sent in sentence_prefixes:    
            sents_encodings.append(encodings(vocab, sent))
        padded_sents = padding(sents_encodings)
        data.append(padded_sents)

    return data


def split_data(X,Y, test_size=0.2, shuffle=True):
    """ Splits data into training and test data by percentages, default test size is 20 %"""
    splits = train_test_split(X, Y, test_size=0.2, shuffle=True)
    X_train, X_test, y_train, y_test = splits

    return X_train, X_test, y_train, y_test

def create_prefixes(sentence):
    """ 
    Creates a hundred instances representing prefixes of one sentence. Returns a list of the prefixes. 
    """
    padded_sents = []
    sent_str = ""
    for s in (sentence):
        sent_str += s
        padded_sents.append(sent_str)

    return padded_sents


def encodings(vocab, sentence):
    """"
    Encoding by mapping each character in a sentence to a correspoding index in a vocabulary.
    """
    encoded = [vocab[ch] for ch in sentence]
    encoded_tensor = torch.LongTensor(encoded)

    return encoded_tensor


def padding(encoded_tensors):
    """ 
    Pads a list of tensors with zeros.
    """
    #print("Length of tensors: ", len(encoded_tensors))
    padded_tensors = pad_sequence(encoded_tensors, batch_first=True, padding_value=0)
    print(padded_tensors)
    return padded_tensors

