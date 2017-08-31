from sklearn.naive_bayes import MultinomialNB
import numpy as np

class SVMClassifier(object):

    def __init__(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

class NBClassifier(object):

    def __init__(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

class DTClassifier(object):

    def __init__(self):
        pass

    def train(self):
        pass

    def test(self):
        pass



if __name__ == '__main__':

    X = np.random.randint(5, size=(6, 100))
    y = np.array([2, 3, -1, 1, 0, 0])

    clf = MultinomialNB()
    clf.fit(X, y)

    test = X[2:3]
    print test
    print clf.predict(test)
    probability =  clf.predict_proba(test)[0]
    print probability

    print clf.classes_

    '''
    Returns the probability of the samples for each class in the model. The columns correspond to the classes in sorted order, as they appear in the attribute classes
    '''

