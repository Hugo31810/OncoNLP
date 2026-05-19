import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

def preprocesamiento(textos, id = 0):
    """Preprocesa una colección de textos utilizando spaCy, aplicando diferentes 
    niveles de limpieza, lematización y filtrado según el identificador indicado.

    Args:
        textos (pandas.Series o numpy.ndarray): Colección de textos a procesar
        id (int, optional): define la estrategia de preprocesamiento a aplicar.
            Por defecto es 0. Las opciones son:
                - 0 (Caso base): Conserva mayúsculas/minúsculas originales. Elimina stopwords y signos de puntuación.
                - 1: Convierte a minúsculas. Elimina stopwords y signos de puntuación.
                - 2: Lematiza y convierte a minúsculas. Elimina stopwords y signos de puntuación.
                - 3: Convierte a minúsculas. Elimina stopwords y signos de puntuación. Conserva 
                    solo adjetivos, verbos, sustantivos y nombres propios (POS tagging).
                - 4: Lematiza y convierte a minúsculas. Elimina stopwords y signos de puntuación. 
                    Conserva solo adjetivos, verbos, sustantivos y nombres propios (POS tagging).

    Returns:
        list: lista de cadenas de texto (`str`), donde cada elemento es el 
        documento original procesado y unido por espacios.
    """
    textos = textos.astype(str).tolist()
    docs = nlp.pipe(textos, disable=["ner"], batch_size=50)
    resultados = []

    for doc in docs:
        match id:
            case 1:
                tokens = [t.text.lower() for t in doc if not t.is_stop and not t.is_punct]
            case 2:
                tokens = [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct]
            case 3:
                tokens = [t.text.lower() for t in doc if not t.is_stop and not t.is_punct 
                        and t.pos_ in ["ADJ", "VERB", "NOUN", "PROPN"]]
            case 4:
                tokens = [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct 
                        and t.pos_ in ["ADJ", "VERB", "NOUN", "PROPN"]]
            case _: # caso base (0)
                tokens = [t.text for t in doc if not t.is_stop and not t.is_punct]

        resultados.append(" ".join(tokens))
        
    return resultados