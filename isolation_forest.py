import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score
import joblib

# SWaT 루트로 작업 폴더 이동
swat_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(swat_root)

def isolation_forest_baseline():
    """Isolation Forest를 사용한 이상탐지 베이스라인"""
    
    print("="*60)
    print("Isolation Forest - Anomaly Detection Baseline")
    print("="*60)
    
    # 1. 정상 데이터로 학습
    df_normal = pd.read_csv('Datasets/normal.csv')
    numeric_df = df_normal.select_dtypes(include=[np.number])
    numeric_df = numeric_df.interpolate(method='linear', limit_direction='both').fillna(0)
    X_normal = numeric_df.values.astype(float)
    
    print(f"\n[Training Data]")
    print(f"  Shape: {X_normal.shape}")
    
    # 2. Isolation Forest 학습
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso_forest.fit(X_normal)
    
    # 정상 데이터 스코어 (anomaly score)
    normal_scores = -iso_forest.score_samples(X_normal)
    print(f"\n[Normal Data Scores]")
    print(f"  Mean: {normal_scores.mean():.6f}")
    print(f"  Std: {normal_scores.std():.6f}")
    print(f"  Min-Max: [{normal_scores.min():.6f}, {normal_scores.max():.6f}]")
    
    # 3. Attack 데이터 평가
    df_attack = pd.read_csv('Datasets/attack.csv')
    numeric_df = df_attack.select_dtypes(include=[np.number])
    numeric_df = numeric_df.interpolate(method='linear', limit_direction='both').fillna(0)
    X_attack = numeric_df.values.astype(float)
    
    attack_scores = -iso_forest.score_samples(X_attack)
    print(f"\n[Attack Data Scores]")
    print(f"  Mean: {attack_scores.mean():.6f}")
    print(f"  Std: {attack_scores.std():.6f}")
    print(f"  Min-Max: [{attack_scores.min():.6f}, {attack_scores.max():.6f}]")
    
    # 4. Merged 데이터 평가
    df_merged = pd.read_csv('Datasets/merged.csv')
    numeric_df = df_merged.select_dtypes(include=[np.number])
    numeric_df = numeric_df.interpolate(method='linear', limit_direction='both').fillna(0)
    X_merged = numeric_df.values.astype(float)
    
    merged_scores = -iso_forest.score_samples(X_merged)
    
    # 레이블 추출
    labels = (df_merged['Normal/Attack'].values == 'Attack').astype(int)
    
    print(f"\n[Merged Data]")
    print(f"  Total samples: {len(merged_scores)}")
    print(f"  Normal: {(labels==0).sum()}, Attack: {(labels==1).sum()}")
    print(f"  Score Mean-Std: {merged_scores.mean():.6f} ± {merged_scores.std():.6f}")
    
    # 5. 임계값 최적화
    print(f"\n{'='*60}")
    print("Threshold Optimization")
    print(f"{'='*60}")
    
    thresholds = {
        'Mean + 1σ': normal_scores.mean() + normal_scores.std(),
        'Mean + 2σ': normal_scores.mean() + 2 * normal_scores.std(),
        'Mean + 3σ': normal_scores.mean() + 3 * normal_scores.std(),
        '95 percentile': np.percentile(normal_scores, 95),
        '99 percentile': np.percentile(normal_scores, 99),
    }
    
    best_f1 = 0
    best_threshold = 0
    
    for method_name, threshold in thresholds.items():
        preds = (merged_scores > threshold).astype(int)
        f1 = f1_score(labels, preds)
        precision = (preds * labels).sum() / max(preds.sum(), 1)
        recall = (preds * labels).sum() / max(labels.sum(), 1)
        
        print(f"\n[{method_name}]")
        print(f"  Threshold: {threshold:.6f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # 6. AUC 계산
    auc = roc_auc_score(labels, merged_scores)
    print(f"\n{'='*60}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Best Threshold: {best_threshold:.6f} (F1={best_f1:.4f})")
    print(f"{'='*60}")
    
    # 7. 최종 평가
    preds = (merged_scores > best_threshold).astype(int)
    print(f"\n[Final Performance]")
    print(classification_report(labels, preds, target_names=['Normal', 'Attack']))
    
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    # 8. 저장
    joblib.dump(iso_forest, 'lstm_ae/isolation_forest_model.pkl')
    np.save('lstm_ae/isolation_forest_scores.npy', merged_scores)
    print(f"\n✓ Model and scores saved")

if __name__ == '__main__':
    isolation_forest_baseline()
