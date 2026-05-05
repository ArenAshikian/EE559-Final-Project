from data_utils import prepare_full_data, make_balanced_subset
from metrics import print_metrics, best_threshold_by_f1
from perceptron import Perceptron
from logistic_regression import LogisticRegressionCustom
from kernel_perceptron import KernelPerceptron
from mlp import MLP
from gaussian_nb import GaussianNaiveBayes
import numpy as np

Xtr,ytr,Xv,yv,Xte,yte = prepare_full_data()
pw = np.sum(ytr==0)/max(np.sum(ytr==1),1)

print("\n=== PERCEPTRON ===")
p=Perceptron()
p.fit(Xtr,ytr)
print_metrics("Perceptron",yte,p.predict(Xte))

print("\n=== LOGISTIC REGRESSION ===")
lr=LogisticRegressionCustom(pos_weight=pw)
lr.fit(Xtr,ytr)
probs=lr.predict_proba(Xv)
t,_=best_threshold_by_f1(yv,probs)
print_metrics("LogReg",yte,lr.predict(Xte,t))

print("\n=== KERNEL PERCEPTRON ===")
Xs,ys=make_balanced_subset(Xtr,ytr)
kp=KernelPerceptron()
kp.fit(Xs,ys)
print_metrics("Kernel",yte,kp.predict(Xte))

print("\n=== MLP ===")
mlp=MLP(Xtr.shape[1],pos_weight=pw)
mlp.fit(Xtr,ytr)
probs=mlp.predict_proba(Xv)
t,_=best_threshold_by_f1(yv,probs)
print_metrics("MLP",yte,mlp.predict(Xte,t))

print("\n=== GAUSSIAN NB ===")
gnb=GaussianNaiveBayes()
gnb.fit(Xtr,ytr)
probs=gnb.predict_proba(Xv)
t,_=best_threshold_by_f1(yv,probs)
print_metrics("GNB",yte,gnb.predict(Xte,t))
