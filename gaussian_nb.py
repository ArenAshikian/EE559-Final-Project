import numpy as np

class GaussianNaiveBayes:
    def fit(self,X,y):
        self.classes=np.unique(y)
        self.mean={}; self.var={}; self.prior={}
        for c in self.classes:
            Xc=X[y==c]
            self.mean[c]=Xc.mean(0)
            self.var[c]=Xc.var(0)+1e-9
            self.prior[c]=len(Xc)/len(X)

    def predict_proba(self,X):
        probs=[]
        for x in X:
            post=[]
            for c in self.classes:
                mean=self.mean[c]; var=self.var[c]
                logp = -0.5*np.sum(np.log(2*np.pi*var)) -0.5*np.sum((x-mean)**2/var)
                post.append(np.log(self.prior[c])+logp)
            post=np.array(post)
            post=np.exp(post - np.max(post))
            post/=post.sum()
            probs.append(post[1])
        return np.array(probs)

    def predict(self,X,t=0.5):
        return (self.predict_proba(X)>=t).astype(int)
