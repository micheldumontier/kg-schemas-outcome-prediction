import sys, time, math, torch, numpy as np, pandas as pd, torch.nn.functional as F
from torch_geometric.nn import RGCNConv
prefix, dev, n = sys.argv[1], torch.device(sys.argv[2]), int(sys.argv[3])
tr = pd.read_csv(f'{prefix}_triples.tsv', sep='\t', header=None)
N = int(max(tr[0].max(), tr[2].max())) + 1; R = int(tr[1].max()) + 1
ei = torch.tensor(np.vstack([tr[2].values, tr[0].values]), dtype=torch.long, device=dev)
et = torch.tensor(tr[1].values, dtype=torch.long, device=dev)
x = torch.randn(N, 32, device=dev)
convs = torch.nn.ModuleList([RGCNConv(32, 32, R, num_bases=8), RGCNConv(32, 32, R, num_bases=8), RGCNConv(32, 3, R, num_bases=8)]).to(dev)
opt = torch.optim.Adam(convs.parameters(), 1e-3); idx = torch.arange(0, 8000, device=dev); y = torch.zeros(8000, dtype=torch.long, device=dev)
def step():
    h = x
    for c in convs[:-1]: h = F.relu(c(h, ei, et))
    out = convs[-1](h, ei, et); loss = F.cross_entropy(out[idx], y); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        h = x
        for c in convs[:-1]: h = F.relu(c(h, ei, et))
        convs[-1](h, ei, et)
step()
if dev.type == 'mps': torch.mps.synchronize()
t = time.time()
for _ in range(n): step()
if dev.type == 'mps': torch.mps.synchronize()
print(f'{prefix} {dev}: {(time.time()-t)/n:.2f} s/epoch (train+eval)')
