import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# SWaT 루트로 작업 폴더 이동 (어디서 실행하든 경로 일관성, Datasets 폴더를 항상 찾을 수 있도록 보장)
swat_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(swat_root)

# 1D 시계열 데이터 → LSTM 입력용 3D 윈도우 변환(행 단위로 데이터를 넣으면 서로 연관없는 독립적인 사건이라고 착각, 윈도우를 만들어주어 시간적 순서를 부여)
def create_windows(data: np.ndarray, window_size: int, step: int):
    n_samples, n_features = data.shape
    windows = []
    for start in range(0, n_samples - window_size + 1, step):
        windows.append(data[start:start + window_size])
    return np.stack(windows)

#csv 파일 처리 + 스케일러 저장(training 데이터용)
def fit_scaler_and_create_windows(csv_path, out_dir, window_size=100, step=50):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    # select numeric columns only
    df = df.select_dtypes(include=[np.number])#불필요한 텍스트 제거
    data = df.values.astype(float)
    scaler = StandardScaler()#정규화, 평균 0, 표준편차 1로 변환
    scaler.fit(data)
    scaled = scaler.transform(data)
    windows = create_windows(scaled, window_size, step)
    np.save(os.path.join(out_dir, "normal_windows.npy"), windows) #학습 데이터
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl")) #추후 evaluation에 사용할 attack.csv를 정규화할 때 normal.csv의 통계를 사용해야 데이터 분포가 일관성 있기 때문에 따로 저장
    print(f"Saved {windows.shape} windows and scaler to {out_dir}")

# 이미 저장된 스케일러를 사용해 다른 csv 정규화(evaluation용)
def transform_file_to_windows(csv_path, scaler_path, out_path, window_size=100, step=50):
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number])
    data = df.values.astype(float)
    scaler = joblib.load(scaler_path)
    scaled = scaler.transform(data) #attack.csv 데이터로 스케일러를 재학습하지 않기 위해 fit()은 하지 않음
    windows = create_windows(scaled, window_size, step)
    np.save(out_path, windows)
    print(f"Saved windows {windows.shape} to {out_path}")

#파일 실행 형식(ex: python preprocess.py --in Datasets/SWaT/SWaT_Dataset_Normal_v1.csv --out lstm_ae/data --window 100 --step 50)
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infile', required=True)
    p.add_argument('--out', dest='outdir', default='lstm_ae/data')
    p.add_argument('--window', type=int, default=100)
    p.add_argument('--step', type=int, default=50)
    args = p.parse_args()
    fit_scaler_and_create_windows(args.infile, args.outdir, args.window, args.step)
