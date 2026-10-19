"""Port of the CARE-SM RGCN training notebook published by the original authors.

Source: https://github.com/TeamHeKA/neurovasc/blob/main/notebooks/Graphs/rgcn_node_pred_care-sm.ipynb
(MIT licence upstream; a copy is in repro/notebooks/). The notebook's code is kept as it is: model, k_fold with
random_state 42 for both StratifiedKFold and train_test_split, training loop with a checkpoint on best validation
accuracy, and the metrics. Every change is marked with a "# CHANGED:" comment. Nothing seeds torch in the notebook,
and none is added here, so runs are not bit-reproducible.
"""
# CHANGED: command-line arguments, results directory and run name replace the notebook's fixed paths
import argparse
import json
import os
import time
import math
import joblib
import numpy as np
import pandas as pd
from statistics import mean, stdev

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, ConfusionMatrixDisplay

import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear

from torch_geometric.data import Data
from torch_geometric.logging import log
from torch_geometric.nn import RGCNConv

# CHANGED: arguments (the notebook has fixed values; defaults below are the notebook's)
ap = argparse.ArgumentParser()
ap.add_argument('--prefix', required=True, help='<prefix>_entities.tsv, _triples.tsv, _numeric.npy')
ap.add_argument('--outcomes', required=True)
ap.add_argument('--num_patients', type=int, default=10000)
ap.add_argument('--layers', type=int, default=3, help='notebook: 3')
ap.add_argument('--emb_std', type=float, default=None, help='default none = notebook xavier start; else N(0, std)')
ap.add_argument('--run', required=True)
ap.add_argument('--results_dir', default='results')
ap.add_argument('--run_folds', type=int, default=None, help='only the first k folds (smoke check)')
ap.add_argument('--epochs', type=int, default=2000, help='notebook: 2000')
args = ap.parse_args()

num_patients = args.num_patients
embed_dim = 100
inverse_triples = False

# CHANGED: paths from arguments instead of processed_data/{graph}_..._{num_patients}
entity = pd.read_csv(f'{args.prefix}_entities.tsv', sep='\t', header=None)
entity = entity.set_index(entity[1])
entity = entity.to_dict()[0]

patients = []
for i in range(num_patients):
    patient = f'<http://nvasc.org/synth_patient_{i}>'
    patients.append(entity[patient])

triples = pd.read_csv(f'{args.prefix}_triples.tsv', sep='\t', header=None)  # CHANGED: path
if inverse_triples == True:
    triples_inv = triples[[2, 1, 0]]
    triples_inv.columns=[0,1,2]
    triples_inv[1] += triples[1].max() + 1
    triples = pd.concat((triples, triples_inv), axis=0)
    triples = triples.reset_index()
else:
    triples_inv = triples[[2, 1, 0]]
    triples_inv.columns=[0,1,2]
    triples = triples_inv
y = np.asarray(joblib.load(args.outcomes))  # CHANGED: path
num_x = torch.Tensor(np.load(f'{args.prefix}_numeric.npy'))  # CHANGED: path

edge_index = torch.vstack((torch.Tensor(triples[0]).long(),torch.Tensor(triples[2]).long()))
edge_type = torch.Tensor(triples[1]).long()
num_nodes = len(entity)

data = Data(
    edge_index=edge_index,
    edge_type=edge_type,
    num_nodes=num_nodes,
)
embedding = Parameter(torch.empty(num_nodes, embed_dim))
torch.nn.init.xavier_uniform_(embedding, gain=math.sqrt(2.0))
if args.emb_std is not None:  # CHANGED: opt-in normal start, still untrained as in the notebook
    torch.nn.init.normal_(embedding, std=args.emb_std)
data.x = embedding
data.num_x=num_x.view(-1,1)
data.num_relations = data.num_edge_types
data.num_classes = 3
print(data)

# CHANGED: results go to <results_dir>/<run>/ (notebook: result/{graph}/rgcnlit)
path = os.path.join(args.results_dir, args.run)
if not os.path.exists(path):
    os.makedirs(path)

def k_fold(X, y, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    train_indices, val_indices, test_indices  = [], [], []
    train_y, val_y, test_y = [], [], []
    for (non_test_idx, test_idx) in skf.split(X, y):
        test_indices.append(X[test_idx])
        train_idx, val_idx, _, _ = train_test_split(non_test_idx, y[non_test_idx], test_size=1/9, random_state=42)
        train_indices.append(X[train_idx])
        val_indices.append(X[val_idx])
        train_y.append(y[train_idx])
        val_y.append(y[val_idx])
        test_y.append(y[test_idx])
    return train_indices, val_indices, test_indices, train_y, val_y, test_y

metrics = []
for fold, (train_idx, val_idx, test_idx, train_y, val_y, test_y) in enumerate(zip(*k_fold(np.asarray(patients), y, folds=10))):
    # CHANGED: smoke-check limit and resumability (finished folds are skipped)
    if args.run_folds is not None and fold >= args.run_folds:
        break
    fold_json = os.path.join(path, f'fold{fold}.json')
    if os.path.exists(fold_json):
        print(f'skip fold {fold} (exists)', flush=True)
        continue

    data.train_idx = torch.Tensor(train_idx).long()
    data.valid_idx = torch.Tensor(val_idx).long()
    data.test_idx = torch.Tensor(test_idx).long()

    data.train_y = torch.Tensor(train_y).long()
    data.valid_y = torch.Tensor(val_y).long()
    data.test_y = torch.Tensor(test_y).long()

    # Train RGCN model.
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_lin = Linear(1, embed_dim)
            self.act_lin = torch.nn.PReLU(embed_dim)
            # CHANGED: layer stack built from --layers; with 3 it is the notebook's conv1 (embed_dim,32), conv2 (32,32),
            # conv3 (32,num_classes) with PReLU(32) after the first two, created in the same order
            dims = [embed_dim] + [32] * (args.layers - 1) + [data.num_classes]
            self.convs = torch.nn.ModuleList(
                RGCNConv(dims[i], dims[i + 1], data.num_relations, num_bases=8) for i in range(args.layers))
            self.acts = torch.nn.ModuleList(torch.nn.PReLU(32) for _ in range(args.layers - 1))

        def forward(self, edge_index, edge_type):
            x = self.act_lin(self.num_lin(data.num_x))
            x = x + data.x
            # CHANGED: loop over layers (notebook: act1(conv1), dropout, act2(conv2), dropout, conv3)
            for conv, act in zip(self.convs[:-1], self.acts):
                x = act(conv(x, edge_index, edge_type))
                x = F.dropout(x, p=0.2, training=self.training)
            x = self.convs[-1](x, edge_index, edge_type)
            return F.log_softmax(x, dim=1)


    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model, data = Net().to(device), data.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=5e-4,
    )

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.edge_index, data.edge_type)
        loss = F.nll_loss(out[data.train_idx], data.train_y)
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test():
        model.eval()
        pred = model(data.edge_index, data.edge_type).argmax(dim=-1)
        train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
        val_acc = float((pred[data.valid_idx] == data.valid_y).float().mean())
        test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
        return train_acc, val_acc, test_acc

    # CHANGED: checkpoint in the results directory, one file per fold
    ckpt = os.path.join(path, f'model_weights_{num_patients}_fold{fold}.pth')
    times = []
    best_val_acc = final_test_acc = 0
    best_epoch = 0  # CHANGED: recorded for the JSON
    for epoch in range(1, args.epochs + 1):  # CHANGED: epochs from argument (default 2000)
        start = time.time()
        loss = train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            best_epoch = epoch  # CHANGED
            torch.save(model.state_dict(), ckpt)  # CHANGED: path
        # CHANGED: print every 100 epochs instead of log() every epoch (keeps the log small)
        if epoch % 100 == 0:
            log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
        times.append(time.time() - start)

    # Evaluation
    model = Net().to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True))  # CHANGED: path
    with torch.no_grad():
        model.eval()
        out = model(data.edge_index, data.edge_type)
        y_pred = out.argmax(dim=-1)
        y_prob = F.softmax(out, dim=-1)
        accuracy = accuracy_score(data.test_y.cpu(), y_pred[data.test_idx].cpu())
        auc_score_class = roc_auc_score(data.test_y.cpu(), y_prob[data.test_idx].cpu(), average=None, multi_class='ovr')
        auc_score_macro = roc_auc_score(data.test_y.cpu(), y_prob[data.test_idx].cpu(), average='macro', multi_class='ovr')
        auc_score = roc_auc_score(data.test_y.cpu(), y_prob[data.test_idx].cpu(), average='weighted', multi_class='ovr')
        precision_class, recall_class, fscore_class, _ = precision_recall_fscore_support(data.test_y.cpu(), y_pred[data.test_idx].cpu(), average=None)
        precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(data.test_y.cpu(), y_pred[data.test_idx].cpu(), average='macro')
        precision, recall, fscore, _ = precision_recall_fscore_support(data.test_y.cpu(), y_pred[data.test_idx].cpu(), average='weighted')

    metric = pd.DataFrame({
    'PRECISION': np.hstack((precision_class, precision_macro, precision)),
    'RECALL': np.hstack((recall_class, recall_macro, recall)),
    'F1SCORE': np.hstack((fscore_class, fscore_macro, fscore)),
    'ACCURACY': np.hstack((np.zeros(4), accuracy)),
    'AUC': np.hstack((auc_score_class, auc_score_macro, auc_score)),
    }, index=['B2H', 'REHAB', 'DEATH', 'MACRO', 'WEIGHTED'])
    metric.index.name = fold
    metrics.append(metric)

    # CHANGED: one JSON per fold with the keys of repro/run_rgcn.py, instead of the notebook's CSV
    r = dict(run=args.run, fold=fold, best_epoch=best_epoch, acc=accuracy,
             auc_weighted=auc_score,
             f1_backhome=fscore_class[0], f1_rehab=fscore_class[1], f1_death=fscore_class[2],
             f1_macro=fscore_macro, f1_weighted=fscore)
    tmp = fold_json + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write(json.dumps(r, default=float))
    os.rename(tmp, fold_json)
    print('RESULT', json.dumps(r, default=float), flush=True)
