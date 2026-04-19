from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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
        self.clf = LogisticRegression(max_iter=500)

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