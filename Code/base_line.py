import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

def base_system(X_train, Y_train, X_test):
    #clase más frecuente en el conjunto de train
    most_frequent_label = Y_train.mode()[0]

    y_pred = [most_frequent_label] * len(X_test)
    return y_pred

def evaluate_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')  



