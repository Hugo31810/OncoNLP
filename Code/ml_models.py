import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture

class MultinomialModel:

    def __init__(self):
        self.clf = MultinomialNB()

    def fit_predict(self, m_train, m_test, y_train):
        # No compatible con embeddings
        self.clf.fit(m_train,y_train)
        y_pred = self.clf.predict(m_test)
        return y_pred
    
    def predict(self, m_test):
        y_pred = self.clf.predict(m_test)
        return y_pred


class GaussianModel:

    def __init__(self):
        self.clf = GaussianNB()


    def fit_predict(self, m_train, m_test, y_train):
        self.clf.fit(m_train, y_train)
        y_pred = self.clf.predict(m_test)
        return y_pred
    
    def predict(self, m_test):
        y_pred = self.clf.predict(m_test)
        return y_pred
    

class LogisticModel:

    def __init__(self):
        self.clf = LogisticRegression(max_iter=200)

    def fit_predict(self, m_train, m_test, y_train):
        self.clf.fit(m_train, y_train)
        y_pred = self.clf.predict(m_test)
        return y_pred
    
    def predict(self, m_test):
        y_pred = self.clf.predict(m_test)
        return y_pred
    
class SVMModel:
    def __init__(self):
        self.clf = SVC(class_weight='balanced')

    def fit_predict(self, m_train, m_test, y_train):
        self.clf.fit(m_train, y_train)
        y_pred = self.clf.predict(m_test)
        return y_pred
    
    def predict(self, m_test):
        y_pred = self.clf.predict(m_test)
        return y_pred

class GMMModel:
    def __init__(self):
        self.models = {}

    def fit(self, X, y):
        self.classes = np.unique(y)

        for cls in self.classes:
            X_c = X[y == cls]
            gmm = GaussianMixture(n_components=1, covariance_type='full')
            self.models[cls] = gmm.fit(X_c)

    def predict(self, X):
        likelihoods = np.array([mod.score_samples(X) for mod in self.models.values()]).T

        return self.classes[np.argmax(likelihoods, axis=1)]