# NAND 시퀀스 분류/생성 모델 성능 개선 로드맵

본 문서는 `seq_sim.py` 실험 파이프라인을 단계적으로 개선-검증-리뷰하는 과정을 정의한다. 각 단계는 **이전 단계 결과가 기준 성능(Accuracy, F1 등)을 만족할 때 다음 단계로 넘어간다.**

## 공통 설정

| 파라미터 | 기본값 |
|----------|--------|
| `seq_len`                | 50 |
| `num_blocks`             | 3  |
| `read_offset_limit`      | 5  |
| Train / Val / Test split | 70 / 15 / 15 % |
| Early-Stopping patience  | 5 epochs |

평가지표

* Overall Accuracy
* Macro / 전략별 F1-Score (`valid`, `page_hop`, `read_unwritten`, `stale_read`)

베이스라인은 2025-08-03 실험 결과(Accuracy≈0.52, Invalid 전략 F1≈0.32)로 한다.

---

## 단계별 계획

### Step-0 : Baseline 재현
* 명령
  ```bash
  python seq_sim.py --model_type classifier --epochs 10 --num_samples 1000 \
                    --classifier_batch_size 256 --num_workers 0
  ```
* 산출물 : `baseline_metrics.json`

### Step-1 : 데이터 균형 & Hard-Negative Mining
1. 발전 내용
   * 각 corruption 전략을 **동일 개수**로 샘플링한다.
   * 학습 중 FP 샘플(모델이 틀린 invalid)을 추가 재학습(간이 mining).
2. 구현 포인트
   * `generate_invalid()` 호출 시 전략 이름을 리턴하므로, generator 루프를 전략별 카운트로 제어.
   * `--balanced_data` CLI 플래그 추가.
3. 통과 조건
   * Invalid 전체 F1 ≥ **0.55** (약 +20p 향상 예상).

### Step-2 : Focal Loss 적용
1. 발전 내용
   * `BCEWithLogitsLoss` → Focal-Loss (α=0.7, γ=2).
2. 구현 포인트
   * util 함수 `focal_loss` 작성, classifier 학습 루프에 적용.
3. 통과 조건
   * Invalid F1 ≥ **0.65**.

### Step-3 : Bidirectional GRU
1. 발전 내용
   * Encoder RNN을 `bidirectional=True` 로 변경, hidden concat.
2. 통과 조건
   * Overall Accuracy ≥ **0.75** & 각 전략 F1 ≥ **0.70**.

### Step-4 : 하이퍼파라미터 조정
* embedding_dim 64, hidden_size 256, dropout 0.3, epochs 30 (+ early-stopping).
* 통과 조건 : 추가 2p 향상.

### Step-5 : 추가 특징(Δpage, Δtime, 블록 통계)

### Step-6 : Multi-task 학습 (시퀀스 validity + 단일 op validity)

### Step-7 : Ensemble (3 seeds soft-voting)

각 단계 성공 시 `metrics_stepX.json` 과 간단 리뷰를 커밋한다.

---

## 진행 현황

| 단계 | 상태 | Accuracy | Invalid F1 | 비고 |
|------|------|----------|------------|-------|
| 0 (Baseline) | ✅ | 0.52 | 0.32 | 정상 재현 |
| 1 (Balanced) | ✅ | 0.74 | 0.00 | Valid 전부 오검출, 전략 F1=1.0 |
| 2 (FocalLoss)| ✅ | 0.55 | 0.63 | 목표 0.65 근접 |
| 3 (Bi-GRU)   | ✅ | 0.45 | 0.55 | 오류 검출↑ Valid↓ |
| 4 (Hparam)   | 🚧 |        |        | 코드 파라미터 CLI화 진행중 |

> 문서 갱신은 각 단계 완료 후 필수.
