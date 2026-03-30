import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

def preprocesamiento(textos, id = 0):
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