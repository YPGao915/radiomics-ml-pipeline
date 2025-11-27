#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# pipeline.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

sns.set(style="whitegrid")

def bootstrap_auc(y_true, y_score, n_bootstraps=2000, seed=42):
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    n = len(y_true)
    for i in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_score[indices])
        bootstrapped_scores.append(score)
    if len(bootstrapped_scores) == 0:
        return (np.nan, np.nan)
    return (np.percentile(bootstrapped_scores, 2.5), np.percentile(bootstrapped_scores, 97.5))

def compute_metrics(y_true, y_score, cutoff=0.5, n_bootstraps=2000):
    auc = roc_auc_score(y_true, y_score)
    auc_ci = bootstrap_auc(np.array(y_true), np.array(y_score), n_bootstraps=n_bootstraps)
    y_pred = (np.array(y_score) >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    f1 = f1_score(y_true, y_pred)
    return {
        'auc': auc,
        'auc_ci': auc_ci,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1
    }

def plot_roc_for_models(models_dict, X, y, outpath):
    plt.figure(figsize=(8,6))
    for name, model in models_dict.items():
        probs = model.predict_proba(X)[:,1]
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0,1],[0,1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_radscore_hist(probas_dict, outpath):
    n = len(probas_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n,4))
    if n == 1:
        axes = [axes]
    for ax, (label, probs) in zip(axes, probas_dict.items()):
        sns.histplot(probs, kde=True, ax=ax)
        ax.set_title(label)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_calibration(models_dict, X, y, outpath):
    plt.figure(figsize=(8,6))
    for name, model in models_dict.items():
        probs = model.predict_proba(X)[:,1]
        prob_true, prob_pred = calibration_curve(y, probs, n_bins=5, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='o', label=name)
    plt.plot([0,1],[0,1], '--', color='gray')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed probability')
    plt.legend()
    plt.savefig(outpath, dpi=300)
    plt.close()

def evaluate_saved_models(models_path, X_test, y_test, final_features, outdir):
    os.makedirs(outdir, exist_ok=True)
    models = joblib.load(os.path.join(models_path, 'all_models.pkl'))
    # models expected: dict name->estimator
    results = {}
    probas = {}
    for name, model in models.items():
        probs = model.predict_proba(X_test[final_features])[:,1]
        results[name] = compute_metrics(y_test, probs)
        probas[name] = probs
    # ROC plot
    plot_roc_for_models(models, X_test[final_features], y_test, os.path.join(outdir, 'roc_all_models.png'))
    # radscore hist
    plot_radscore_hist(probas, os.path.join(outdir, 'radscore_distribution.png'))
    # calibration
    plot_calibration(models, X_test[final_features], y_test, os.path.join(outdir, 'calibration.png'))
    return results

if __name__ == '__main__':
    print('Pipeline module. Edit paths below or call functions from your notebook/script.')
    # Example usage (uncomment and edit paths):
    # models_path = './stage_Integrated_saved_models'
    # X_test = pd.read_csv('holdout_test.csv')  # must contain final_features columns
    # y_test = X_test['status'].values
    # final_features = joblib.load(os.path.join(models_path, 'selected_features.pkl'))
    # results = evaluate_saved_models(models_path, X_test, y_test, final_features, './results')
    # print(results)

