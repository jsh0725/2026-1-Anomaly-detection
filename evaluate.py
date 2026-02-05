import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from model import LSTMAE

# SWaT 루트로 작업 폴더 이동
swat_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(swat_root)


def reconstruction_errors(model, data, device):
    model.eval()
    # 입력 데이터 상태 확인
    print(f"Input data shape: {data.shape}")
    print(f"Input data dtype: {data.dtype}")
    print(f"Input data stats: min={np.nanmin(data):.6f}, max={np.nanmax(data):.6f}, mean={np.nanmean(data):.6f}")
    print(f"NaN count: {np.sum(np.isnan(data))}, Inf count: {np.sum(np.isinf(data))}")
    
    # NaN/Inf를 0으로 치환
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print("WARNING: Replacing NaN/Inf with 0...")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 배치 처리 (메모리 절약)
    batch_size = 64
    all_errs = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            x = torch.from_numpy(batch).float().to(device)
            recon = model(x).cpu().numpy()
            
            # per-window MSE
            batch_errs = np.mean((recon - batch) ** 2, axis=(1, 2))
            all_errs.extend(batch_errs)
    
    errs = np.array(all_errs)
    # 재구성 출력 상태 확인
    print(f"Error stats: min={np.min(errs):.6f}, max={np.max(errs):.6f}, mean={np.mean(errs):.6f}")
    return errs


def evaluate(model_path, data_path, labels_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(data_path)
    n_features = data.shape[2]
    model = LSTMAE(n_features)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)  # ← 모델을 device로 이동
    model.eval()  # 평가 모드 설정
    errs = reconstruction_errors(model, data, device)
    print(f"Errors: mean={errs.mean():.6f}, max={errs.max():.6f}")
    
    # 오차를 자동 저장
    save_path = data_path.replace('.npy', '_errs.npy')
    np.save(save_path, errs)
    print(f"Saved errors to {save_path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--labels', required=False)
    args = p.parse_args()
    evaluate(args.model, args.data, args.labels)
