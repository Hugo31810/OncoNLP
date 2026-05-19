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
    """Transforma una colección de textos en representaciones vectoriales numéricas 
    (embeddings) utilizando un modelo de palabras (como Word2Vec, GloVe o FastText).

    Args:
        textos (pandas.Series o numpy.ndarray): Colección de textos a vectorizar (debe soportar .astype(str))
        modelo (gensim.models, dict, o array-like): Modelo de word embeddings.
        metodo (int, optional): Define la estrategia de agregación. Por defecto es 0.
        - 0 (Caso base): Calcula el promedio (mean pooling).
        - 1: Calcula la suma (sum pooling).
        - 2: Sin agregación. Devuelve la secuencia completa de vectores.

    Returns:
        numpy.ndarray: array que contiene las representaciones vectoriales
    """
    textos = textos.astype(str).tolist()
    matrix = []
    for texto in textos:
        if hasattr(modelo, 'wv'):
            palabras = [w for w in texto.split() if w in modelo.wv]
        else:
            palabras = [w for w in texto.split() if w in modelo]

        if palabras:

            if hasattr(modelo, 'wv'):
                vectores = modelo.wv[palabras]
            else:
                vectores = modelo[palabras]

            match metodo:
                case 1:
                    vector = np.sum(vectores, axis = 0)
                case 2:
                    vector = vectores            
                case _:
                    vector = np.mean(vectores, axis=0)
        
        matrix.append(vector)

    return np.array(matrix)