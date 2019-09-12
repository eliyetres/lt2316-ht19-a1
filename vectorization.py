from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

def make_vectors(x_data, y_data):
    vectorizer = TfidfVectorizer(lowercase=False)
    x_data = vectorizer.fit_transform(x_data)
    y_data = vectorizer.fit_transform(y_data)

    X = x_data.todense()
    X = X.tolist()
    X = np.asarray(X)