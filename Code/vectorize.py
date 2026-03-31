from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def vectorizacionBinaria(textos, max_vocab = None):
    textos = textos.astype(str).tolist()
    vect = CountVectorizer(binary=True, max_features=max_vocab)
    matrix = vect.fit_transform(textos)
    return matrix, vect

def vectorizacionFreq(textos, max_vocab = None):
    textos = textos.astype(str).tolist()
    vect = CountVectorizer(max_features=max_vocab)
    matrix = vect.fit_transform(textos)
    return matrix, vect

def vectorizacionTfidf(textos, max_vocab = None):
    textos = textos.astype(str).tolist()
    vect = TfidfVectorizer(max_features=max_vocab)
    matrix = vect.fit_transform(textos)
    return matrix, vect

def vectorizacionEmbeddings(textos, modelo, metodo = 0):
    textos = textos.astype(str).tolist()
    matrix = []
    for texto in textos:
        palabras = [w for w in texto.split() if w in modelo]

        if palabras:
            match metodo:
                case 1:
                    vector = np.sum(modelo[palabras], axis = 0)
                case 2:
                    vector = modelo[palabras]            
                case _:
                    vector = np.mean(modelo[palabras], axis = 0)
        
        matrix.append(vector)

    return np.array(matrix)