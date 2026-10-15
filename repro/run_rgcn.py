"""RGCN+lit runner, ported from neurovasc/exp/models/node_pred_rgcn_sphn.py.

Kept identical: entity/triple loading, edge reversal (object -> subject only),
fixed random xavier node features (not registered as a model parameter, so never
trained), Linear(1,d)+PReLU literal encoder added to node features, RGCNConv with
num_bases=8, PReLU, dropout, Adam(lr, wd), full-batch training, best-val-accuracy
checkpoint selection, StratifiedKFold(random_state=77) + 1/9 validation split,
weighted one-vs-rest AUC.

Generalised: input files and number of RGCN layers (3 = paper's RGCN3+lit,
5 = RGCN5+lit), number of epochs, one JSON file per fold (results/<run>/fold<k>.json),
resumable (finished folds are skipped), per-fold seed = --seed + fold.
"""
import argparse, json, math, time, copy
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from pathlib import Path
from splits import paper_splits
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument('--prefix', required=True, help='<prefix>_entities.tsv, _triples.tsv, _numeric.npy')
ap.add_argument('--outcomes', required=True)
ap.add_argument('--num_patients', type=int, default=10000)
ap.add_argument('--layers', type=int, default=3)
ap.add_argument('--folds', type=int, default=10)
ap.add_argument('--run_folds', type=int, default=None, help='only run the first k folds')
ap.add_argument('--epochs', type=int, default=2000)
ap.add_argument('--dr', type=float, default=0.2)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--wd', type=float, default=5e-4)
ap.add_argument('--embed_dim', type=int, default=32)
ap.add_argument('--hidden_dim', type=int, default=32)
ap.add_argument('--device', default='cpu')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--both_directions', action='store_true', help='also pass messages subject to object, as separate relation types')
ap.add_argument('--init_emb', default=None, help='npy [num_nodes, embed_dim]; non-zero rows replace the initial node embedding')
ap.add_argument('--zero_emb_regex', default=None, help='entities whose name matches start with a zero embedding (e.g. patient-specific nodes)')
ap.add_argument('--emb_std', type=float, default=None, help='diagnostic: init node embeddings N(0, std) instead of xavier (still untrained)')
ap.add_argument('--run', required=True, help='run name; results go to <results_dir>/<run>/fold<k>.json')
ap.add_argument('--results_dir', default='results')
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)

entity = pd.read_csv(f'{args.prefix}_entities.tsv', sep='\t', header=None)
names = [None] * len(entity)
for i_, n_ in zip(entity[0], entity[1]):
    names[int(i_)] = n_
entity = entity.set_index(entity[1])
entity = entity.to_dict()[0]
patients = [entity[f'<http://nvasc.org/synth_patient_{i}>'] for i in range(args.num_patients)]

triples = pd.read_csv(f'{args.prefix}_triples.tsv', sep='\t', header=None)
triples_inv = triples[[2, 1, 0]]
triples_inv.columns = [0, 1, 2]
triples = triples_inv
y = np.asarray(joblib.load(args.outcomes))
num_x = torch.Tensor(np.load(f'{args.prefix}_numeric.npy'))

edge_index = torch.vstack((torch.Tensor(triples[0].values).long(), torch.Tensor(triples[2].values).long()))
edge_type = torch.Tensor(triples[1].values).long()
if args.both_directions:   # add the original subject->object direction as new relation types
    n_rel = int(edge_type.max()) + 1
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_type = torch.cat([edge_type, edge_type + n_rel])
num_nodes = len(entity)
data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes)
embedding = Parameter(torch.empty(num_nodes, args.embed_dim))
torch.nn.init.xavier_uniform_(embedding, gain=math.sqrt(2.0))
if args.emb_std is not None:
    torch.nn.init.normal_(embedding, std=args.emb_std)
from emb_mask import zero_mask
_mask = torch.tensor(zero_mask(names, args.zero_emb_regex))
with torch.no_grad():
    embedding[_mask] = 0.0
print('zeroed embeddings:', int(_mask.sum()), flush=True)
if args.init_emb:
    _init = torch.tensor(np.load(args.init_emb), dtype=torch.float32)
    assert _init.shape == embedding.shape, f'init_emb shape {tuple(_init.shape)} != {tuple(embedding.shape)}'
    _rows = _init.abs().sum(1) > 0
    with torch.no_grad():
        embedding[_rows] = _init[_rows]
    print('initial embeddings loaded for', int(_rows.sum()), 'nodes', flush=True)
data.x = embedding
data.num_x = num_x.view(-1, 1)
data.num_relations = int(edge_type.max()) + 1
data.num_classes = 3
print(data, 'nonzero literals:', int((num_x != 0).sum()), flush=True)


class Net(torch.nn.Module):
    def __init__(self, L):
        super().__init__()
        d, h = args.embed_dim, args.hidden_dim
        self.num_lin = Linear(1, d)
        self.act_lin = torch.nn.PReLU(d)
        dims = [d] + [h] * (L - 1) + [data.num_classes]
        self.convs = torch.nn.ModuleList(RGCNConv(dims[i], dims[i + 1], data.num_relations, num_bases=8) for i in range(L))
        self.acts = torch.nn.ModuleList(torch.nn.PReLU(h) for _ in range(L - 1))

    def forward(self, edge_index, edge_type):
        x = self.act_lin(self.num_lin(data.num_x))
        x = x + data.x
        for conv, act in zip(self.convs[:-1], self.acts):
            x = act(conv(x, edge_index, edge_type))
            x = F.dropout(x, p=args.dr, training=self.training)
        x = self.convs[-1](x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


device = torch.device(args.device)
data = data.to(device)
out_dir = Path(args.results_dir) / args.run
out_dir.mkdir(parents=True, exist_ok=True)
patients = np.asarray(patients)
for fold, (tr_i, va_i, te_i) in enumerate(paper_splits(y)):
    if args.run_folds is not None and fold >= args.run_folds:
        break
    path = out_dir / f'fold{fold}.json'
    if path.exists():
        print(f'skip fold {fold} (exists)', flush=True)
        continue
    torch.manual_seed(args.seed + fold); np.random.seed(args.seed + fold)
    tr, va, te = (torch.tensor(patients[i], device=device) for i in (tr_i, va_i, te_i))
    ytr, yva, yte = (torch.tensor(y[i], device=device) for i in (tr_i, va_i, te_i))
    model = Net(args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val, best_state, best_epoch = 0, None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        out = model(data.edge_index, data.edge_type)
        loss = F.nll_loss(out[tr], ytr); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(data.edge_index, data.edge_type).argmax(-1)
        val_acc = float((pred[va] == yva).float().mean())
        if val_acc > best_val:
            best_val, best_state, best_epoch = val_acc, copy.deepcopy(model.state_dict()), epoch
        if epoch % 100 == 0:
            print(f'fold {fold} epoch {epoch} loss {float(loss):.4f} train {float((pred[tr]==ytr).float().mean()):.3f} '
                  f'val {val_acc:.3f} best {best_val:.3f}@{best_epoch} {time.time()-t0:.0f}s', flush=True)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        out = model(data.edge_index, data.edge_type)
    prob = F.softmax(out, -1)[te].cpu().numpy(); yp = prob.argmax(1); yt = yte.cpu().numpy()
    _, _, f1c, _ = precision_recall_fscore_support(yt, yp, average=None, labels=[0, 1, 2], zero_division=0)
    r = dict(run=args.run, fold=fold, best_epoch=best_epoch, acc=accuracy_score(yt, yp),
             auc_weighted=roc_auc_score(yt, prob, average='weighted', multi_class='ovr'),
             f1_backhome=f1c[0], f1_rehab=f1c[1], f1_death=f1c[2],
             f1_macro=precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)[2],
             f1_weighted=precision_recall_fscore_support(yt, yp, average='weighted', zero_division=0)[2],
             pred_class_counts=np.bincount(yp, minlength=3).tolist())
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(r, default=float))
    tmp.rename(path)          # atomic, so an interrupted fold never leaves a partial file
    print('RESULT', json.dumps(r, default=float), flush=True)
