import numpy as np

class KernelPerceptron:
    def __init__(self, epochs=10, gamma=0.01):
        self.epochs=epochs; self.gamma=gamma

    def rbf(self,X1,X2):
        return np.exp(-self.gamma*(np.sum(X1**2,1)[:,None]+np.sum(X2**2,1)-2*X1@X2.T))

    def fit(self,X,y):
        self.X=X
        self.y=np.where(y==1,1,-1)
        self.a=np.zeros(len(X))
        K=self.rbf(X,X)
        for _ in range(self.epochs):
            for i in range(len(X)):
                s=np.sum(self.a*self.y*K[:,i])
                if np.sign(s)!=self.y[i]:
                    self.a[i]+=1

    def predict(self,X):
        K=self.rbf(self.X,X)
        s=np.sum((self.a*self.y)[:,None]*K,axis=0)
        return (s>=0).astype(int)
