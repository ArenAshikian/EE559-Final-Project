import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-500,500)))

class MLP:
    def __init__(self, inp, h=64, lr=0.0001, epochs=30, pos_weight=1):
        self.lr=lr; self.epochs=epochs; self.pos_weight=pos_weight
        self.W1=np.random.randn(inp,h)*0.01
        self.b1=np.zeros((1,h))
        self.W2=np.random.randn(h,1)*0.01
        self.b2=np.zeros((1,1))

    def relu(self,x): return np.maximum(0,x)

    def fit(self,X,y):
        y=y.reshape(-1,1); m=len(X)
        for _ in range(self.epochs):
            z1=X@self.W1+self.b1; a1=self.relu(z1)
            z2=a1@self.W2+self.b2; a2=sigmoid(z2)
            wts=np.where(y==1,self.pos_weight,1)
            dz2=wts*(a2-y)/m
            dW2=a1.T@dz2; db2=np.sum(dz2,0)
            da1=dz2@self.W2.T; dz1=da1*(z1>0)
            dW1=X.T@dz1; db1=np.sum(dz1,0)
            self.W1-=self.lr*dW1; self.b1-=self.lr*db1
            self.W2-=self.lr*dW2; self.b2-=self.lr*db2

    def predict_proba(self,X):
        return sigmoid(self.relu(X@self.W1+self.b1)@self.W2+self.b2).ravel()

    def predict(self,X,t=0.5):
        return (self.predict_proba(X)>=t).astype(int)
