"""LR, RF, the paper's NN (100/50/10, tanh) and an age-only depth-2 tree, on the graph pipeline's CSV and folds."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from splits import paper_splits

AGE_COL = None  # column index of age, set by load_table
EVENT_COLS = None  # column indices of the 8 care events, set by load_table
EVENTS = ['nimodipine', 'paracetamol', 'nad', 'corotrop', 'morphine', 'dve', 'atl', 'iot']


def load_table(csv):
    global AGE_COL, EVENT_COLS
    df = pd.read_csv(csv, index_col=0).rename(columns={'output': 'outcome'})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)      # graph patient i == row i
    y = df.outcome.astype(int).values
    X = df.drop(columns=['outcome'])
    AGE_COL = list(X.columns).index('age')
    EVENT_COLS = [list(X.columns).index(e) for e in EVENTS]
    return X.values, y, list(X.columns)


def age_only(X):
    return X[:, [AGE_COL]]


def events_only(X):
    return X[:, EVENT_COLS]


MODELS = {
    'LR': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    'RF': lambda: RandomForestClassifier(100, random_state=0, n_jobs=-1),
    'NN': lambda: make_pipeline(StandardScaler(), MLPClassifier((100, 50, 10), activation='tanh', max_iter=2000,
                                                                 early_stopping=True, random_state=0)),
    'AGE': lambda: make_pipeline(FunctionTransformer(age_only), DecisionTreeClassifier(max_depth=2, random_state=0)),
    'AGE_RF': lambda: make_pipeline(FunctionTransformer(age_only), RandomForestClassifier(100, random_state=0, n_jobs=-1)),
    'EVENTS_RF': lambda: make_pipeline(FunctionTransformer(events_only), RandomForestClassifier(100, random_state=0, n_jobs=-1)),
}


def metrics(y_true, prob):
    yp = prob.argmax(1)
    f1c = precision_recall_fscore_support(y_true, yp, average=None, labels=[0, 1, 2], zero_division=0)[2]
    return dict(best_epoch=0, acc=accuracy_score(y_true, yp),
                auc_weighted=roc_auc_score(y_true, prob, average='weighted', multi_class='ovr'),
                f1_backhome=f1c[0], f1_rehab=f1c[1], f1_death=f1c[2],
                f1_macro=precision_recall_fscore_support(y_true, yp, average='macro', zero_division=0)[2],
                f1_weighted=precision_recall_fscore_support(y_true, yp, average='weighted', zero_division=0)[2],
                pred_class_counts=np.bincount(yp, minlength=3).tolist())


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--models', default=','.join(MODELS), help='comma-separated subset of ' + ','.join(MODELS))
    ap.add_argument('--suffix', default='', help='appended to each result directory name, e.g. _SEPT2024')
    ap.add_argument('--results_dir', default='results')
    args = ap.parse_args()
    X, y, cols = load_table(args.csv)
    for name in args.models.split(','):
        make = MODELS[name]
        out = Path(args.results_dir) / (name + args.suffix)
        out.mkdir(parents=True, exist_ok=True)
        for k, (tr, va, te) in enumerate(paper_splits(y)):
            m = make().fit(X[tr], y[tr])
            rec = dict(run=name, fold=k, **metrics(y[te], m.predict_proba(X[te])))
            (out / f'fold{k}.json').write_text(json.dumps(rec, default=float))
        print(name, 'done', flush=True)
