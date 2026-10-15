"""Tabular RF / LR on the same CSV, patient order and fold split the graph runner uses."""
import sys, json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
df = pd.read_csv(sys.argv[1], index_col=0).rename(columns={'output': 'outcome'})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # same order as graph patient ids
y = df.outcome.astype(int).values
assert list(y) == joblib.load(sys.argv[2]), 'patient order mismatch with graph outcomes'
X = df.drop(columns=['outcome']).values
res = {'RF': [], 'LR': []}
for k, (non_test, test) in enumerate(StratifiedKFold(10, shuffle=True, random_state=77).split(X, y)):
    train, val, _, _ = train_test_split(non_test, y[non_test], test_size=1/9, random_state=77)
    for name, m in [('RF', RandomForestClassifier(100, n_jobs=4, random_state=0)),
                    ('LR', make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)))]:
        m.fit(X[train], y[train]); p = m.predict_proba(X[test])
        res[name].append((roc_auc_score(y[test], p, average='weighted', multi_class='ovr'), accuracy_score(y[test], p.argmax(1))))
for name, r in res.items():
    r = np.array(r)
    print(f'{name}: fold0 AUC={r[0,0]:.3f} acc={r[0,1]:.3f} | 10-fold AUC={r[:,0].mean():.3f}±{r[:,0].std():.3f} acc={r[:,1].mean():.3f}±{r[:,1].std():.3f}')
