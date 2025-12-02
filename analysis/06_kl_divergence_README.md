# KL Divergence Analysis (06_kl_divergence.py)

## 개요

Base 모델과 SFT/Our 모델 간의 KL divergence를 측정하여, Our 모델이 `<IDK>` 응답을 할 때와 하지 않을 때의 분포 차이를 분석합니다.

## 목적

- Our 모델이 "모른다"(`<IDK>`)고 답할 때, base 모델과의 분포 차이가 어떻게 변하는지 확인
- SFT 모델과 비교하여 Our 모델의 학습 효과 분석

## 분석 흐름

```
1. Our 모델로 inference → <IDK> 응답 여부 판단
2. Gold answer에 대해 KL divergence 계산:
   - KL(Base || SFT)
   - KL(Base || Our)
3. <IDK> 그룹 / Non-<IDK> 그룹별 통계 비교
```

## 사용법

```bash
# 기본 실행
python 06_kl_divergence.py --dataname halueval --our_model_name sample_weighted_reverse_llm_idk0.2

# 테스트 (샘플 수 제한)
python 06_kl_divergence.py --dataname halueval --our_model_name sample_weighted_reverse_llm_idk0.2 --max_samples 100

# 다른 데이터셋
python 06_kl_divergence.py --dataname medqa --our_model_name sample_weighted_reverse_llm_idk0.2
python 06_kl_divergence.py --dataname sciq --our_model_name sample_weighted_reverse_llm_idk0.2
```

## 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--dataname` | 데이터셋 (halueval, medqa, sciq) | 필수 |
| `--our_model_name` | Our 모델 이름 | 필수 |
| `--base_model` | Base 모델 경로 | `../../model/Llama-3.2-3B` |
| `--max_samples` | 최대 샘플 수 | None (전체) |
| `--output_dir` | 출력 디렉토리 | `kl_divergence` |

## 출력

### 파일 위치
```
kl_divergence/{dataname}/kl_analysis_{our_model_name}.json
```

### 출력 구조

```json
{
  "statistics": {
    "kl_base_sft_all_count": 100,
    "kl_base_sft_all_mean": 1.23,
    "kl_base_sft_all_std": 0.45,
    ...
    "kl_base_our_idk_count": 30,
    "kl_base_our_idk_mean": 2.15,
    ...
    "kl_base_our_no_idk_count": 70,
    "kl_base_our_no_idk_mean": 1.05,
    ...
    "idk_count": 30,
    "total_count": 100,
    "idk_ratio": 0.30,
    "idk_vs_no_idk_diff": 1.10,
    "our_vs_sft_on_idk_samples": 0.85,
    "our_vs_sft_on_no_idk_samples": -0.15
  },
  "samples": [
    {
      "question": "...",
      "prompt": "Question: ...\n\nAnswer:",
      "gold_response": "...",
      "our_answer": "... <IDK><|end_of_text|>",
      "is_idk": true,
      "kl_base_sft": 1.23,
      "kl_base_our": 2.08,
      "kl_diff": 0.85
    },
    ...
  ]
}
```

## 통계 설명

| 통계 | 설명 |
|------|------|
| `kl_base_sft_all` | 전체 샘플에 대한 KL(Base \|\| SFT) |
| `kl_base_our_all` | 전체 샘플에 대한 KL(Base \|\| Our) |
| `kl_base_our_idk` | Our 모델이 `<IDK>` 응답한 샘플의 KL(Base \|\| Our) |
| `kl_base_our_no_idk` | Our 모델이 `<IDK>` 없이 응답한 샘플의 KL(Base \|\| Our) |
| `kl_base_sft_idk` | Our 모델이 `<IDK>` 응답한 샘플의 KL(Base \|\| SFT) |
| `kl_base_sft_no_idk` | Our 모델이 `<IDK>` 없이 응답한 샘플의 KL(Base \|\| SFT) |

### 비교 메트릭

| 메트릭 | 의미 |
|--------|------|
| `idk_vs_no_idk_diff` | Our 모델의 `<IDK>` 샘플 KL - Non-`<IDK>` 샘플 KL |
| `our_vs_sft_on_idk_samples` | `<IDK>` 샘플에서 Our KL - SFT KL |
| `our_vs_sft_on_no_idk_samples` | Non-`<IDK>` 샘플에서 Our KL - SFT KL |

## 기술적 세부사항

### 토크나이저 분리

- **base_tokenizer**: Base/SFT 모델용, KL 계산용
- **our_tokenizer**: Our 모델 inference용 (`<IDK>` 토큰 포함)

### Vocab Size 불일치 처리

Our 모델은 `<IDK>` 토큰이 추가되어 vocab size가 +1 됩니다 (128256 → 128257).
KL 계산 시 공통 vocab size로 truncate하여 비교합니다.

```python
min_vocab_size = min(logits1.shape[-1], logits2.shape[-1])
logits1 = logits1[:, :, :min_vocab_size]
logits2 = logits2[:, :, :min_vocab_size]
```

### 프롬프트 형식

`01_eval.py`와 동일한 형식 사용:
```
Question: {question}

Answer:
```

### KL Divergence 계산

Gold answer 토큰에 대해 계산:
```
KL(P_base || P_target) = Σ P_base(x) * log(P_base(x) / P_target(x))
```

## 예상 결과 해석

- `idk_vs_no_idk_diff > 0`: Our 모델이 `<IDK>` 응답할 때 base와 더 다름 → 불확실할 때 분포가 크게 변함
- `our_vs_sft_on_idk_samples > 0`: `<IDK>` 샘플에서 Our가 SFT보다 base와 더 다름
- `our_vs_sft_on_no_idk_samples < 0`: Non-`<IDK>` 샘플에서 Our가 SFT보다 base와 비슷함 → 확신 있는 답변은 base 유지
