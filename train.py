import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

from model import LSTMAE

# SWaT 루트로 작업 폴더 이동
swat_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(swat_root)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(args.data)
    # data shape: (n_windows, seq_len, n_features)
    dataset = TensorDataset(torch.from_numpy(data).float())
    dl = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    seq_len = data.shape[1]
    n_features = data.shape[2]
    model = LSTMAE(n_features, hidden_size=args.hidden, latent_size=args.latent, num_layers=1).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in dl:
            batch = batch.to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * batch.size(0)
        print(f"Epoch {epoch}/{args.epochs} loss={total / len(dataset):.6f}")
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.out, f"model_ep{epoch}.pt"))
    torch.save(model.state_dict(), os.path.join(args.out, "model_final.pt"))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='lstm_ae/checkpoints')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--latent', type=int, default=16)
    p.add_argument('--save-every', type=int, default=5)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train(args)
