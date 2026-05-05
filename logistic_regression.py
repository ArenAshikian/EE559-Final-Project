import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-500,500)))

class LogisticRegressionCustom:
    def __init__(self, lr=0.01, epochs=100, pos_weight=1):
        self.lr=lr; self.epochs=epochs; self.pos_weight=pos_weight

    def fit(self,X,y):
        self.w = np.zeros(X.shape[1]); self.b=0
        m=len(X)
        for _ in range(self.epochs):
            p = sigmoid(X@self.w + self.b)
            wts = np.where(y==1,self.pos_weight,1)
            err = wts*(p-y)
            self.w -= self.lr*(X.T@err/m)
            self.b -= self.lr*np.sum(err)/m

    def predict_proba(self,X):
        return sigmoid(X@self.w + self.b)

    def predict(self,X,t=0.5):
        return (self.predict_proba(X)>=t).astype(int)
