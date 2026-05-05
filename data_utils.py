import numpy as np
import pandas as pd

def load_data(csv_path="creditcard.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop("Class", axis=1).values.astype(float)
    y = df["Class"].values.astype(int)
    return X, y

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X, y = X[idx], y[idx]
    n = len(X)
    t = int(train_ratio*n)
    v = int((train_ratio+val_ratio)*n)
    return X[:t], y[:t], X[t:v], y[t:v], X[v:], y[v:]

def standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1
    return mean, std

def standardize_transform(X, m, s):
    return (X - m) / s

def prepare_full_data(path="creditcard.csv"):
    X,y = load_data(path)
    Xtr,ytr,Xv,yv,Xte,yte = train_val_test_split(X,y)
    m,s = standardize_fit(Xtr)
    return standardize_transform(Xtr,m,s),ytr,standardize_transform(Xv,m,s),yv,standardize_transform(Xte,m,s),yte

def make_balanced_subset(X,y,mult=3):
    pos = np.where(y==1)[0]
    neg = np.where(y==0)[0]
    keep_neg = np.random.choice(neg, size=min(len(neg), mult*len(pos)), replace=False)
    idx = np.concatenate([pos, keep_neg])
    np.random.shuffle(idx)
    return X[idx], y[idx]
