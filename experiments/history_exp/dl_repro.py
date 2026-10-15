"""Minimal repro: does DataLoader with num_workers>0 hang on this setup?"""
import os
import sys
import time

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DS(Dataset):
    def __init__(self, n=800_000):
        self.x = np.random.randint(0, 100, (n, 8))
        self.y = np.random.rand(n, 2)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


def run(num_workers, epochs=2, use_cuda=True):
    ds = DS()
    dl = DataLoader(ds, batch_size=40960, shuffle=True, num_workers=num_workers)
    dev = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"
    lin = torch.nn.Linear(8, 2).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    t0 = time.time()
    for e in range(epochs):
        nb = 0
        for x, y in dl:
            x, y = x.float().to(dev), y.float().to(dev)
            loss = ((lin(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            nb += 1
        print(f"  workers={num_workers} epoch {e}: {nb} batches, {time.time()-t0:.1f}s", flush=True)
    print(f"workers={num_workers} OK ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    nw = int(sys.argv[1])
    dev = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    run(nw, use_cuda=(dev == "cuda"))
