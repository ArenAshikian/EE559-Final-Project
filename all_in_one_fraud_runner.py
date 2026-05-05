
import json
import numpy as np
from pathlib import Path

# =========================
# Data utils
# =========================
def load_data(csv_path="creditcard.csv"):
    import pandas as pd
    df = pd.read_csv(csv_path)
    X = df.drop("Class", axis=1).values.astype(float)
    y = df["Class"].values.astype(int)
    return X, y


def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]

    n = len(X)
    t = int(train_ratio * n)
    v = int((train_ratio + val_ratio) * n)

    return X[:t], y[:t], X[t:v], y[t:v], X[v:], y[v:]


def standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def standardize_transform(X, mean, std):
    return (X - mean) / std


def prepare_full_data(csv_path="creditcard.csv", seed=42):
    X, y = load_data(csv_path)
    Xtr, ytr, Xv, yv, Xte, yte = train_val_test_split(X, y, seed=seed)
    mean, std = standardize_fit(Xtr)
    Xtr = standardize_transform(Xtr, mean, std)
    Xv = standardize_transform(Xv, mean, std)
    Xte = standardize_transform(Xte, mean, std)
    return Xtr, ytr, Xv, yv, Xte, yte


def make_balanced_subset(X, y, negative_multiplier=3, seed=42):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    num_neg_keep = min(len(neg_idx), negative_multiplier * len(pos_idx))
    sampled_neg = rng.choice(neg_idx, size=num_neg_keep, replace=False)
    idx = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(idx)
    return X[idx], y[idx]


# =========================
# Metrics
# =========================
def compute_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)

    return acc, prec, rec, f1, tp, tn, fp, fn


def best_threshold_by_f1(y_true, probs):
    best_t = 0.5
    best_f1 = -1.0

    for t in np.arange(0.01, 0.99, 0.01):
        preds = (probs >= t).astype(int)
        _, _, _, f1, _, _, _, _ = compute_metrics(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


# =========================
# Models
# =========================
class Perceptron:
    def __init__(self, lr=0.001, epochs=20):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        y_s = np.where(y == 1, 1, -1)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        for epoch in range(self.epochs):
            mistakes = 0
            for i in range(len(X)):
                pred = 1 if X[i] @ self.w + self.b >= 0 else -1
                if pred != y_s[i]:
                    self.w += self.lr * y_s[i] * X[i]
                    self.b += self.lr * y_s[i]
                    mistakes += 1
            print(f"Perceptron epoch {epoch + 1}/{self.epochs} - mistakes: {mistakes}")

    def predict(self, X):
        return (X @ self.w + self.b >= 0).astype(int)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegressionCustom:
    def __init__(self, lr=0.01, epochs=100, pos_weight=1.0):
        self.lr = lr
        self.epochs = epochs
        self.pos_weight = pos_weight

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        m = len(X)

        for epoch in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            weights = np.where(y == 1, self.pos_weight, 1.0)
            err = weights * (p - y)

            self.w -= self.lr * (X.T @ err / m)
            self.b -= self.lr * np.sum(err) / m

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                loss = -np.mean(
                    weights * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
                )
                print(f"LogReg epoch {epoch + 1}/{self.epochs} - loss: {loss:.6f}")

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, t=0.5):
        return (self.predict_proba(X) >= t).astype(int)


class KernelPerceptron:
    def __init__(self, epochs=10, gamma=0.01):
        self.epochs = epochs
        self.gamma = gamma

    def rbf(self, X1, X2):
        return np.exp(
            -self.gamma * (
                np.sum(X1 ** 2, axis=1)[:, None]
                + np.sum(X2 ** 2, axis=1)
                - 2 * X1 @ X2.T
            )
        )

    def fit(self, X, y):
        self.X = X
        self.y = np.where(y == 1, 1, -1)
        self.a = np.zeros(len(X))
        K = self.rbf(X, X)

        for epoch in range(self.epochs):
            mistakes = 0
            for i in range(len(X)):
                s = np.sum(self.a * self.y * K[:, i])
                pred = 1 if s >= 0 else -1
                if pred != self.y[i]:
                    self.a[i] += 1
                    mistakes += 1
            print(f"Kernel epoch {epoch + 1}/{self.epochs} - mistakes: {mistakes}")

    def predict(self, X):
        K = self.rbf(self.X, X)
        s = np.sum((self.a * self.y)[:, None] * K, axis=0)
        return (s >= 0).astype(int)


class MLP:
    def __init__(self, inp, h=64, lr=0.0001, epochs=30, pos_weight=1.0):
        self.lr = lr
        self.epochs = epochs
        self.pos_weight = pos_weight

        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.01, size=(inp, h))
        self.b1 = np.zeros((1, h))
        self.W2 = rng.normal(0, 0.01, size=(h, 1))
        self.b2 = np.zeros((1, 1))

    def relu(self, x):
        return np.maximum(0.0, x)

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        m = len(X)

        for epoch in range(self.epochs):
            z1 = X @ self.W1 + self.b1
            a1 = self.relu(z1)
            z2 = a1 @ self.W2 + self.b2
            a2 = sigmoid(z2)

            weights = np.where(y == 1, self.pos_weight, 1.0)

            dz2 = weights * (a2 - y) / m
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                loss = -np.mean(
                    weights * (y * np.log(a2 + 1e-12) + (1 - y) * np.log(1 - a2 + 1e-12))
                )
                print(f"MLP epoch {epoch + 1}/{self.epochs} - loss: {loss:.6f}")

    def predict_proba(self, X):
        a1 = self.relu(X @ self.W1 + self.b1)
        return sigmoid(a1 @ self.W2 + self.b2).ravel()

    def predict(self, X, t=0.5):
        return (self.predict_proba(X) >= t).astype(int)


class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}

        for c in self.classes:
            Xc = X[y == c]
            self.mean[c] = Xc.mean(axis=0)
            self.var[c] = Xc.var(axis=0) + 1e-9
            self.prior[c] = len(Xc) / len(X)

    def predict_proba(self, X):
        probs = []

        for x in X:
            post = []
            for c in self.classes:
                mean = self.mean[c]
                var = self.var[c]
                logp = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum((x - mean) ** 2 / var)
                post.append(np.log(self.prior[c] + 1e-12) + logp)

            post = np.array(post)
            post = np.exp(post - np.max(post))
            post /= post.sum()
            probs.append(post[1])

        return np.array(probs)

    def predict(self, X, t=0.5):
        return (self.predict_proba(X) >= t).astype(int)


# =========================
# HTML report
# =========================
def build_results_entry(y_true, y_pred, notes=""):
    acc, p, r, f1, tp, tn, fp, fn = compute_metrics(y_true, y_pred)
    return {
        "metrics": {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        },
        "notes": notes,
    }


def confusion_html(metrics):
    return f"""
    <div class="cm-grid">
      <div class="cm-box tp"><div class="cm-label">TP</div><div class="cm-value">{metrics["tp"]}</div></div>
      <div class="cm-box fp"><div class="cm-label">FP</div><div class="cm-value">{metrics["fp"]}</div></div>
      <div class="cm-box fn"><div class="cm-label">FN</div><div class="cm-value">{metrics["fn"]}</div></div>
      <div class="cm-box tn"><div class="cm-label">TN</div><div class="cm-value">{metrics["tn"]}</div></div>
    </div>
    """


def bar_group_html(title, metric_key, results):
    max_val = max(model["metrics"][metric_key] for model in results.values())
    max_val = max(max_val, 1e-9)

    rows = []
    for model_name, model_data in results.items():
        value = model_data["metrics"][metric_key]
        pct = 100.0 * value / max_val
        rows.append(f"""
        <div class="bar-row">
          <div class="bar-label">{model_name}</div>
          <div class="bar-wrap"><div class="bar-fill" style="width:{pct:.2f}%"></div></div>
          <div class="bar-value">{value:.4f}</div>
        </div>
        """)

    return f"""
    <section class="card">
      <h2>{title}</h2>
      {''.join(rows)}
    </section>
    """


def build_html(results):
    rows = []
    for model_name, model_data in results.items():
        m = model_data["metrics"]
        notes = model_data.get("notes", "")
        rows.append(f"""
        <tr>
          <td>{model_name}</td>
          <td>{m["accuracy"]:.4f}</td>
          <td>{m["precision"]:.4f}</td>
          <td>{m["recall"]:.4f}</td>
          <td>{m["f1"]:.4f}</td>
          <td>{m["tp"]}</td>
          <td>{m["tn"]}</td>
          <td>{m["fp"]}</td>
          <td>{m["fn"]}</td>
          <td>{notes}</td>
        </tr>
        """)

    f1_winner = max(results.items(), key=lambda x: x[1]["metrics"]["f1"])
    recall_winner = max(results.items(), key=lambda x: x[1]["metrics"]["recall"])
    precision_winner = max(results.items(), key=lambda x: x[1]["metrics"]["precision"])

    confusion_sections = []
    for model_name, model_data in results.items():
        confusion_sections.append(f"""
        <section class="card">
          <h2>{model_name} Confusion Summary</h2>
          {confusion_html(model_data["metrics"])}
        </section>
        """)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fraud Detection Analysis Report</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
      background: #f6f8fb;
      color: #17202a;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    .hero, .card {{
      background: white;
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.06);
      margin-bottom: 24px;
    }}
    .hero p {{
      margin: 8px 0;
      line-height: 1.5;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .stat {{
      background: #f9fbff;
      border: 1px solid #e2e8f0;
      border-radius: 14px;
      padding: 16px;
    }}
    .label {{
      font-size: 13px;
      color: #5b6777;
      margin-bottom: 8px;
    }}
    .value {{
      font-size: 20px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid #e6ebf2;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f8fafc;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 190px 1fr 80px;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .bar-label {{
      font-size: 14px;
      font-weight: 600;
    }}
    .bar-wrap {{
      height: 18px;
      background: #eef2f7;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, #4f46e5, #06b6d4);
      border-radius: 999px;
    }}
    .bar-value {{
      font-size: 14px;
      text-align: right;
      font-weight: 600;
    }}
    .cm-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(110px, 1fr));
      gap: 12px;
      max-width: 360px;
    }}
    .cm-box {{
      border-radius: 14px;
      padding: 18px 16px;
      color: white;
    }}
    .tp {{ background: #16a34a; }}
    .tn {{ background: #2563eb; }}
    .fp {{ background: #dc2626; }}
    .fn {{ background: #d97706; }}
    .cm-label {{
      font-size: 13px;
      opacity: 0.9;
    }}
    .cm-value {{
      font-size: 26px;
      font-weight: 700;
      margin-top: 4px;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 24px;
    }}
    code {{
      background: #eef2f7;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <h1>Fraud Detection Analysis Report</h1>
      <p>This page summarizes the latest run of all five models and generates a clean comparison you can use for your final report and presentation.</p>
      <div class="summary-grid">
        <div class="stat">
          <div class="label">Best F1 Score</div>
          <div class="value">{f1_winner[0]} — {f1_winner[1]["metrics"]["f1"]:.4f}</div>
        </div>
        <div class="stat">
          <div class="label">Best Recall</div>
          <div class="value">{recall_winner[0]} — {recall_winner[1]["metrics"]["recall"]:.4f}</div>
        </div>
        <div class="stat">
          <div class="label">Best Precision</div>
          <div class="value">{precision_winner[0]} — {precision_winner[1]["metrics"]["precision"]:.4f}</div>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Model Comparison Table</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>TP</th>
            <th>TN</th>
            <th>FP</th>
            <th>FN</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>

    {bar_group_html("Precision Comparison", "precision", results)}
    {bar_group_html("Recall Comparison", "recall", results)}
    {bar_group_html("F1 Score Comparison", "f1", results)}

    <div class="two-col">
      {''.join(confusion_sections)}
    </div>
  </div>
</body>
</html>"""


def save_outputs(results, json_path="results.json", html_path="analysis_report.html"):
    Path(json_path).write_text(json.dumps(results, indent=2))
    Path(html_path).write_text(build_html(results), encoding="utf-8")


# =========================
# Main
# =========================
def main():
    Xtr, ytr, Xv, yv, Xte, yte = prepare_full_data("creditcard.csv")
    pos_weight = np.sum(ytr == 0) / max(np.sum(ytr == 1), 1)

    results = {}

    print("\n=== PERCEPTRON ===")
    perc = Perceptron(lr=0.001, epochs=20)
    perc.fit(Xtr, ytr)
    results["Perceptron"] = build_results_entry(
        yte,
        perc.predict(Xte),
        notes="Simple linear baseline with hard updates."
    )

    print("\n=== LOGISTIC REGRESSION ===")
    lr = LogisticRegressionCustom(lr=0.01, epochs=100, pos_weight=pos_weight)
    lr.fit(Xtr, ytr)
    probs_lr = lr.predict_proba(Xv)
    best_t_lr, best_f1_lr = best_threshold_by_f1(yv, probs_lr)
    results["Logistic Regression"] = build_results_entry(
        yte,
        lr.predict(Xte, best_t_lr),
        notes=f"Best validation threshold = {best_t_lr:.2f}, validation F1 = {best_f1_lr:.4f}."
    )

    print("\n=== KERNEL PERCEPTRON ===")
    Xk, yk = make_balanced_subset(Xtr, ytr, negative_multiplier=3, seed=42)
    kp = KernelPerceptron(epochs=10, gamma=0.01)
    kp.fit(Xk, yk)
    results["Kernel Perceptron"] = build_results_entry(
        yte,
        kp.predict(Xte),
        notes="Used a balanced subset to keep kernel computation manageable."
    )

    print("\n=== MLP ===")
    mlp = MLP(Xtr.shape[1], h=64, lr=0.0001, epochs=30, pos_weight=pos_weight)
    mlp.fit(Xtr, ytr)
    probs_mlp = mlp.predict_proba(Xv)
    best_t_mlp, best_f1_mlp = best_threshold_by_f1(yv, probs_mlp)
    results["MLP"] = build_results_entry(
        yte,
        mlp.predict(Xte, best_t_mlp),
        notes=f"Best validation threshold = {best_t_mlp:.2f}, validation F1 = {best_f1_mlp:.4f}."
    )

    print("\n=== GAUSSIAN NAIVE BAYES ===")
    gnb = GaussianNaiveBayes()
    gnb.fit(Xtr, ytr)
    probs_gnb = gnb.predict_proba(Xv)
    best_t_gnb, best_f1_gnb = best_threshold_by_f1(yv, probs_gnb)
    results["Gaussian Naive Bayes"] = build_results_entry(
        yte,
        gnb.predict(Xte, best_t_gnb),
        notes=f"Best validation threshold = {best_t_gnb:.2f}, validation F1 = {best_f1_gnb:.4f}."
    )

    print("\n=== SUMMARY ===")
    for model_name, model_data in results.items():
        m = model_data["metrics"]
        print(f"{model_name}: Acc={m['accuracy']:.4f} Prec={m['precision']:.4f} Rec={m['recall']:.4f} F1={m['f1']:.4f}")

    save_outputs(results)
    print("\nSaved:")
    print("  results.json")
    print("  analysis_report.html")
    print("\nOpen analysis_report.html in your browser.")


if __name__ == "__main__":
    main()
