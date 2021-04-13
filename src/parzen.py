import numpy as np
from tqdm import tqdm

class ParzenWindowEstimate:
    def __init__(self, h, kernel):
        self.h      = h
        self.kernel = kernel

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.labels  = np.unique(self.y_train)

    def p_x(self, x, training_data):
        n    = len(training_data)
        V_n  = self.h
        K_n  = 0
        for x_i in training_data:
            K_n += self.kernel((x - x_i)/self.h)

        return (K_n/n)/V_n

    def predict(self, X_test, verbose = False):
        y_pred = []
        if not verbose:
            for x in X_test:
                likelihoods = [self.p_x(x, self.X_train[self.y_train == Y_, :]) for Y_ in self.labels]
                y_pred.append(self.labels[np.argmax(likelihoods)])
        else:
            for x in tqdm(X_test):
                likelihoods = [self.p_x(x, self.X_train[self.y_train == Y_, :]) for Y_ in self.labels]
                y_pred.append(self.labels[np.argmax(likelihoods)])
        return np.array(y_pred)