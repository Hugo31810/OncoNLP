import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

def base_system(X_train, Y_train, X_test):
    """Implementa un sistema base (baseline) de clasificación que predice siempre 
    la clase mayoritaria (la más frecuente) observada en los datos de entrenamiento.

    Args:
        X_train (pandas.DataFrame o array-like): vector de características de entrenamiento
        Y_train (pandas.Series): vector de etiquetas de entrenamiento
        X_test (pandas.DataFrame o array-like): vector de características de test

    Returns:
        list: lista de longitud igual a X_test donde todos los valores son la etiqueta más común de Y_train
    """
    most_frequent_label = Y_train.mode()[0]

    y_pred = [most_frequent_label] * len(X_test)
    return y_pred

def evaluate_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')  



