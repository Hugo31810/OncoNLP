from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def multinomial_fit_predict(m_train, m_test, y_train):
    # No compatible con embeddings
    clf = MultinomialNB()
    clf.fit(m_train,y_train)
    y_pred = clf.predict(m_test)
    return y_pred

def gaussian_fit_predict(m_train, m_test, y_train):
    clf = GaussianNB()
    clf.fit(m_train, y_train)
    y_pred = clf.predict(m_test)
    return y_pred

def logistic_regression_fit_predict(m_train, m_test, y_train):
    clf = LogisticRegression()
    clf.fit(m_train, y_train)
    y_pred = clf.predict(m_test)
    return y_pred

def svm_fit_predict(m_train, m_test, y_train):
    clf = SVC(class_weight='balanced')
    clf.fit(m_train, y_train)
    y_pred = clf.predict(m_test)
    return y_pred