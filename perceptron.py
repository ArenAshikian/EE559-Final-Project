import numpy as np

class Perceptron:
    def __init__(self, lr=0.001, epochs=20):
        self.lr=lr; self.epochs=epochs

    def fit(self,X,y):
        y_s = np.where(y==1,1,-1)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.epochs):
            for i in range(len(X)):
                pred = 1 if X[i]@self.w + self.b >=0 else -1
                if pred != y_s[i]:
                    self.w += self.lr*y_s[i]*X[i]
                    self.b += self.lr*y_s[i]

    def predict(self,X):
        return (X@self.w + self.b >=0).astype(int)
