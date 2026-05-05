import numpy as np

def compute_metrics(y_true, y_pred):
    tp = np.sum((y_true==1)&(y_pred==1))
    tn = np.sum((y_true==0)&(y_pred==0))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    acc = (tp+tn)/(tp+tn+fp+fn+1e-12)
    prec = tp/(tp+fp+1e-12)
    rec = tp/(tp+fn+1e-12)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    return acc,prec,rec,f1,tp,tn,fp,fn

def print_metrics(name,y_true,y_pred):
    acc,p,r,f1,tp,tn,fp,fn = compute_metrics(y_true,y_pred)
    print(f"\n{name}")
    print(f"Acc:{acc:.4f} Prec:{p:.4f} Rec:{r:.4f} F1:{f1:.4f}")
    print(f"TP:{tp} TN:{tn} FP:{fp} FN:{fn}")

def best_threshold_by_f1(y_true, probs):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.01,0.99,0.01):
        preds = (probs>=t).astype(int)
        _,_,_,f1,_,_,_,_ = compute_metrics(y_true,preds)
        if f1>best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1
