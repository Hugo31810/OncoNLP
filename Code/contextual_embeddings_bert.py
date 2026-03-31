import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# cargamos el tokenizer y el modelo preentrenado de BERT
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# modo evaluación - no queremos entrenar, queremos obtener los embeddings
model.eval()

# función para obtener los embeddings de un texto
def get_token_embeddings(text, max_length=256):
    tokens = tokenizer(text, return_tensors='pt', padding='True', truncation=True, max_length=max_length) #tokenizamos el text y lo convertimos a tensores de PyTorch

    with torch.no_grad(): #no queremos calcular gradientes, solo obtener los embeddings
        outputs = model(**tokens) #obtenemos los outputs del modelo

    embeddigns = outputs.last_hidden_state.numpy() #obtenemos los embeddings de la última capa oculta y los convertimos a numpy
    return embeddigns[:, 1:-1, :] #eliminamos los tokens especiales [CLS] y [SEP]

'''
BERT devuelve un vector por token.
Pero para clasificar necesitamos un único vector por texto.
Para ello haremos Pooling sobre los vectores de los tokens.
'''
def bert_mean_embedding(text, max_length=256):
    embeddigns = get_token_embeddings(text, max_length)
    mean_vector = np.mean(embeddigns, axis=1) #hacemos pooling por media sobre los vectores de los tokens
    return mean_vector[0] #devolvemos el vector resultante

def bert_max_embedding(text, max_length=256):
    embeddigns = get_token_embeddings(text, max_length)
    max_vector = np.max(embeddigns, axis=1) #hacemos pooling por max sobre los vectores de los tokens
    return max_vector[0] #devolvemos el vector resultante


def bert_embedding(texts, pooling='mean', max_length=256):
    if pooling == 'mean':
        return np.array([bert_mean_embedding(text, max_length) for text in texts])
    elif pooling == 'max':
        return np.array([bert_max_embedding(text, max_length) for text in texts])
    else:
        raise ValueError("Pooling debe ser 'mean' o 'max'")