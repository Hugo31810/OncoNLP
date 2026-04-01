import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

#cargamos el modelo preentrenado de ELMo desde TensorFlow Hub
elmo = hub.KerasLayer(
    "https://tfhub.dev/google/elmo/3",
    signature="default",
    output_key="elmo",
    trainable=False,
    dtype=tf.string
)
#funcion para obtener los embeddings contextualizados token por token de un texto usando ELMo
def get_token_embeddings(text):
    embeddings = elmo(tf.constant([text])).numpy()
    return embeddings

def elmo_mean_embedding(text):
    embeddings = get_token_embeddings(text)
    mean_vector = np.mean(embeddings, axis =1) #hacemgos pooling por media sobre los vectores de los tokens
    return mean_vector[0] #devolvemos el vector resultante

def elmo_max_embedding(text):
    embeddings = get_token_embeddings(text)
    max_vector = np.max(embeddings, axis=1) #hacemos pooling por max sobre los vectores de los tokens
    return max_vector[0] #devolvemos el vector resultante   

def elmo_embedding(texts, pooling="mean"):
    texts = texts.astype(str).tolist() if hasattr(texts, "astype") else list(texts)

    if pooling == "mean":
        return np.array([elmo_mean_embedding(text) for text in texts])
    elif pooling == "max":
        return np.array([elmo_max_embedding(text) for text in texts])
    else:
        raise ValueError("Pooling debe ser 'mean' o 'max'")