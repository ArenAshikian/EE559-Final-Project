# EE559 Final Project: Credit Card Fraud Detection

This project compares several supervised learning models for credit card fraud detection. The main goal is to study how different models behave on a highly imbalanced classification problem, where fraudulent transactions are rare and accuracy alone can be misleading.

All models are implemented from scratch using NumPy instead of machine learning libraries.

## Dataset

The project uses the Kaggle Credit Card Fraud Detection dataset. Each transaction is labeled as either non-fraudulent (`0`) or fraudulent (`1`). Most of the input features are anonymized PCA-transformed variables (`V1` through `V28`), along with transaction information such as `Time` and `Amount`.

Because the dataset is highly imbalanced, model performance is evaluated mainly using precision, recall, and F1-score rather than accuracy alone.

To run the project, download `creditcard.csv` from Kaggle and place it in the main project folder.

Expected location:

```text
creditcard.csv
```

## Models Implemented

The following models are implemented manually using NumPy:

- Perceptron
- Logistic Regression
- Kernel Perceptron with RBF kernel
- One-hidden-layer MLP
- Gaussian Naive Bayes

## Project Files

```text
all_in_one_fraud_runner.py   # Runs the full project in one script
run_experiments.py           # Runs model experiments

data_utils.py                # Data loading, splitting, and standardization
metrics.py                   # Accuracy, precision, recall, F1-score, confusion counts

perceptron.py                # Perceptron model
logistic_regression.py       # Logistic regression model
kernel_perceptron.py         # Kernel perceptron model
mlp.py                       # One-hidden-layer neural network
gaussian_nb.py               # Gaussian Naive Bayes model

results.json                 # Final experiment results
analysis_report.html         # Generated HTML summary report
requirements.txt             # Python dependencies
```

## How to Run

Create and activate a virtual environment if desired, then install the required packages:

```bash
pip install -r requirements.txt
```

Make sure `creditcard.csv` is in the same folder as the Python files.

Run the full project:

```bash
python all_in_one_fraud_runner.py
```

Or run the experiment script:

```bash
python run_experiments.py
```

The code produces model results and saves summary outputs such as:

```text
results.json
analysis_report.html
```

## Final Results

The best-performing model was Logistic Regression, which achieved the strongest F1-score among the tested models.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Perceptron | 0.9988 | 0.7288 | 0.5375 | 0.6187 |
| Logistic Regression | 0.9992 | 0.7683 | 0.7875 | 0.7778 |
| Kernel Perceptron | 0.9827 | 0.0878 | 0.8750 | 0.1596 |
| MLP | 0.0019 | 0.0019 | 1.0000 | 0.0037 |
| Gaussian Naive Bayes | 0.9796 | 0.0696 | 0.8000 | 0.1280 |

Logistic Regression performed best because class weighting and threshold tuning helped it balance precision and recall. The Kernel Perceptron and Gaussian Naive Bayes models had high recall but low precision, meaning they detected many fraud cases but also produced many false positives. The MLP struggled with the severe class imbalance and tended to overpredict the fraud class.

## Conclusion

This project shows that for highly imbalanced fraud detection, the most complex model is not always the best model. A simpler Logistic Regression model with class weighting and validation-based threshold tuning produced the best overall F1-score. Future improvements could include stronger regularization, more systematic hyperparameter tuning, resampling methods, and additional neural network architectures.
