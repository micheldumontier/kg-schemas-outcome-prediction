"""Calibration metrics for three-class predicted probabilities (opt-in --calibration)."""
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def calibration_metrics(y_true, prob, n_classes=3):
    """Multiclass Brier score (mean over patients of the sum over classes of (p - one-hot)^2), log loss, macro OvR AUC."""
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=float)
    onehot = np.eye(n_classes)[y_true]
    return dict(brier=float(((prob - onehot) ** 2).sum(1).mean()),
                log_loss=float(log_loss(y_true, prob, labels=list(range(n_classes)))),
                auc_macro=float(roc_auc_score(y_true, prob, average='macro', multi_class='ovr')))

def save_probs(root, run, fold, test_idx, y_true, prob):
    """Write patient-level test predictions of one fold to <root>/<run>/fold<k>.npz (stays on the secure machine)."""
    from pathlib import Path
    d = Path(root) / run
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f'fold{fold}.tmp.npz'
    np.savez(tmp, idx=np.asarray(test_idx), y=np.asarray(y_true), prob=np.asarray(prob, dtype=float))
    tmp.rename(d / f'fold{fold}.npz')


def platt_recalibrate(prob_val, y_val, prob_test, n_classes=3, eps=1e-6):
    """One-vs-rest sigmoid (Platt) recalibration: per class, logistic regression of the class indicator on
    logit(p) fitted on the validation predictions, applied to the test predictions, then rows renormalised."""
    from sklearn.linear_model import LogisticRegression
    logit = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    out = np.zeros((len(prob_test), n_classes))
    for k in range(n_classes):
        m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(prob_val[:, [k]]), (np.asarray(y_val) == k).astype(int))
        out[:, k] = m.predict_proba(logit(prob_test[:, [k]]))[:, 1]
    return out / out.sum(1, keepdims=True)
