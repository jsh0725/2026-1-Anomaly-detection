import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# SWaT 루트로 작업 폴더 이동
swat_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(swat_root)

def analyze_anomaly_detection():
    """LSTM-AE 이상탐지 성능 분석"""
    
    # 재구성 오차 로드
    print("="*60)
    print("LSTM-AE 이상탐지 성능 분석")
    print("="*60)
    
    # 1. 정상 데이터 오차로 임계값 결정
    normal_errs = np.load('lstm_ae/data/normal_windows_errs.npy')
    print(f"\n[Normal Data]")
    print(f"  샘플 수: {len(normal_errs)}")
    print(f"  평균: {normal_errs.mean():.2f}")
    print(f"  표준편차: {normal_errs.std():.2f}")
    print(f"  중앙값: {np.median(normal_errs):.2f}")
    print(f"  Min-Max: [{normal_errs.min():.2f}, {normal_errs.max():.2f}]")
    
    # 2. 공격 데이터 오차
    attack_errs = np.load('lstm_ae/data/attack_windows_errs.npy')
    print(f"\n[Attack Data]")
    print(f"  샘플 수: {len(attack_errs)}")
    print(f"  평균: {attack_errs.mean():.2f}")
    print(f"  표준편차: {attack_errs.std():.2f}")
    print(f"  중앙값: {np.median(attack_errs):.2f}")
    print(f"  Min-Max: [{attack_errs.min():.2f}, {attack_errs.max():.2f}]")
    
    # 3. merged 데이터 오차와 레이블
    merged_errs = np.load('lstm_ae/data/merged_windows_errs.npy')
    df_merged = pd.read_csv('Datasets/merged.csv')
    
    # 레이블 추출 (윈도우 크기=100, 스텝=50으로 다운샘플)
    labels = []
    window_size = 100
    step = 50
    for start in range(0, len(df_merged) - window_size + 1, step):
        window_labels = df_merged['Normal/Attack'].iloc[start:start+window_size].values
        # 윈도우 내 "Attack"이 하나라도 있으면 이상
        label = 1 if np.any(window_labels == 'Attack') else 0
        labels.append(label)
    
    labels = np.array(labels)
    print(f"\n[Merged Data (정상+공격)]")
    print(f"  샘플 수: {len(merged_errs)}")
    print(f"  정상: {(labels==0).sum()}, 공격: {(labels==1).sum()}")
    print(f"  평균: {merged_errs.mean():.2f}")
    print(f"  Min-Max: [{merged_errs.min():.2f}, {merged_errs.max():.2f}]")
    
    # 4. 임계값 결정 (방법별)
    print(f"\n{'='*60}")
    print("임계값 설정 (정상 데이터 통계 기반)")
    print(f"{'='*60}")
    
    methods = {
        'Mean + 1σ': normal_errs.mean() + normal_errs.std(),
        'Mean + 2σ': normal_errs.mean() + 2 * normal_errs.std(),
        'Mean + 3σ': normal_errs.mean() + 3 * normal_errs.std(),
        '95 percentile': np.percentile(normal_errs, 95),
        '99 percentile': np.percentile(normal_errs, 99),
    }
    
    best_f1 = 0
    best_threshold = 0
    
    for method_name, threshold in methods.items():
        preds = (merged_errs > threshold).astype(int)
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        print(f"\n[{method_name}]")
        print(f"  임계값: {threshold:.2f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # 5. 최적 임계값
    print(f"\n{'='*60}")
    print(f"최적 임계값: {best_threshold:.2f} (F1={best_f1:.4f})")
    print(f"{'='*60}")
    
    # 6. AUC 계산
    auc = roc_auc_score(labels, merged_errs)
    print(f"\nAUC-ROC: {auc:.4f}")
    
    # 7. 최적 임계값으로 최종 평가
    preds = (merged_errs > best_threshold).astype(int)
    print(f"\n{'='*60}")
    print("최종 성능 (최적 임계값)")
    print(f"{'='*60}")
    print(classification_report(labels, preds, target_names=['Normal', 'Attack']))
    
    # 혼동 행렬
    cm = confusion_matrix(labels, preds)
    print(f"\n혼동 행렬:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    # 8. 결과 저장
    results = {
        'normal_mean': normal_errs.mean(),
        'normal_std': normal_errs.std(),
        'attack_mean': attack_errs.mean(),
        'threshold': best_threshold,
        'auc': auc,
        'f1': f1_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
    }
    
    print(f"\n{'='*60}")
    print("요약")
    print(f"{'='*60}")
    print(f"정상 평균 오차: {results['normal_mean']:.2f}")
    print(f"공격 평균 오차: {results['attack_mean']:.2f}")
    print(f"임계값: {results['threshold']:.2f}")
    print(f"AUC: {results['auc']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    
    # ROC 곡선 시각화 (선택사항)
    try:
        fpr, tpr, _ = roc_curve(labels, merged_errs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC={auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - LSTM-AE Anomaly Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('lstm_ae/results_roc.png')
        print(f"\n✓ ROC 곡선 저장: lstm_ae/results_roc.png")
    except Exception as e:
        print(f"\n⚠ ROC 곡선 생성 실패: {e}")

if __name__ == '__main__':
    analyze_anomaly_detection()
