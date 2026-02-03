import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from lstm_ae.model import LSTMAE


def reconstruction_errors(model, data, device):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(data).float().to(device)
        recon = model(x).cpu().numpy()
    # per-window MSE
    errs = np.mean((recon - data) ** 2, axis=(1, 2))
    return errs


def evaluate(model_path, data_path, labels_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(data_path)
    n_features = data.shape[2]
    model = LSTMAE(n_features)
    model.load_state_dict(torch.load(model_path, map_location=device))
    errs = reconstruction_errors(model, data, device)
    print(f"Errors: mean={errs.mean():.6f}, max={errs.max():.6f}")
    if labels_path:
        labels = np.load(labels_path)
        auc = roc_auc_score(labels, errs)
        print(f"AUC: {auc:.4f}")
    np.save(data_path.replace('.npy', '_errs.npy'), errs)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--labels', required=False)
    args = p.parse_args()
    evaluate(args.model, args.data, args.labels)
