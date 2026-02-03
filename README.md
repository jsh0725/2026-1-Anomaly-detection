# LSTM-AE for SWaT

간단한 LSTM 기반 오토인코더(재구성 기반 이상탐지) 파이프라인 스캐폴드입니다.

사용 예시:

1. 전처리(정규화 및 윈도잉):

```bash
python lstm_ae/preprocess.py --in Datasets/normal.csv --out lstm_ae/data --window 100 --step 50
```

2. 학습:

```bash
python lstm_ae/train.py --data lstm_ae/data/normal_windows.npy --out lstm_ae/checkpoints
```

3. 평가:

```bash
python lstm_ae/evaluate.py --model lstm_ae/checkpoints/model_final.pt --data lstm_ae/data/merged_windows.npy
```

필요 시 `attack.csv`/`merged.csv`에 대해 `preprocess.transform_file_to_windows`를 사용해 윈도우를 생성하고 평가하세요.
