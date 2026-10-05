# C5 재랭커 v2 계약 (질의·후보 임베딩 상호작용 MLP)

초안: 2026-10-05 (메인). 이슈 #20. 근거: v0·v1(유사도 스칼라 특징)은 모의 라벨 100,000개로도 검증셋 macro R@5가
0.858 근처에 머물렀다(PR #19) — 병목은 라벨 수가 아니라 특징. 사용자 승인(권장안 1).
학습 곡선(#11)·비교 평가(#10)를 보기 전에 고정한다. 사전등록 기준(목표 0.887, 측정 지점)은 바꾸지 않는다.

## 1. 범위
- v0와 같다: 순위 파일 상위 20만 다시 매기고 21–100위는 그대로. 정답 필드는 읽지 않는다.

## 2. 입력 벡터
- q = 질의 임베딩(캐시, 매니페스트로 query_id → image_sha), g = 후보 상품의 갤러리 이미지 중 q와 내적이 가장 큰 이미지의 벡터. 둘 다 L2 정규화.
- 벡터는 `pr eval`과 같은 index·같은 임베딩 캐시에서 읽는다(`pr cand-stats`와 같은 경로). 벡터를 읽는 index가 통계 파일의 index_id와 다르면 실패한다.
- 모델의 `index_id`는 **학습 갤러리**다. 다른 분할(검증셋)을 재정렬할 때는 그 분할의 index를 쓰므로 모델의 index_id와 같을 필요가 없다
  (분할마다 갤러리가 다르다). 재정렬 때는 통계 파일 행들이 하나의 index_id를 갖는지만 확인한다. (2026-10-05 실데이터 실행에서 발견해 수정)
- 입력 x = [q ⊙ g, |q − g|, v1 특징 10개(표준화)] — 768 + 768 + 10 = 1,546차원(임베딩 768 기준).

## 3. 모델·학습 (사전 고정)
- MLP: 1,546 → 256 (ReLU, dropout 0.2) → 1. 출력 = 로짓.
- 손실: 이진 교차 엔트로피, sample_weight = 라벨 weight. Adam lr 1e-3, weight_decay 1e-4, batch 256.
- 조기 종료: 라벨을 query_id 해시로 80/20 나눈 학습 내부 홀드아웃(sha256(seed, query_id) 기준)에서 손실이 10 epoch 동안 나아지지 않으면 멈추고 최저 손실 시점 가중치를 쓴다. 최대 200 epoch.
  **검증셋(val)은 학습·조기 종료·하이퍼파라미터 선택에 쓰지 않는다.**
- 홀드아웃에 pos나 neg가 없으면(라벨이 적을 때) 조기 종료 없이 고정 50 epoch을 쓰고 그렇게 기록한다.
- 결정성: CPU, `torch.manual_seed(seed)`, `torch.use_deterministic_algorithms(True)`. seed는 c4-v3의 seed(0, 1, 2)와 같은 값을 쓴다.
- 라벨 쌍이 상위 20 밖이면 실패, pos·neg 중 하나라도 0개면 실패(v0와 같음).

## 4. 버전·저장
- 계약 문자열 "c5-rerank-v2". 버전 = sha256(계약, label_version, 순위 파일 sha256, 통계 파일 sha256, index_id, 하이퍼파라미터, seed, torch 버전)[:12].
- `artifacts/rerank/{version}/`: `model.json`(설정·표준화 값·학습 기록: epoch 수, 홀드아웃 손실, 조기 종료 여부) + `weights.safetensors` 또는 `weights.pt`(state_dict만).
  같은 버전이 있으면 덮지 않고 같은지 확인한다.

## 5. 추론·출력
- v0와 같은 출력 형식(`rerank_scores`, `reranker_version`), 동점은 원래 순위.
- CLI: `pr train-rerank --version v2 --config <cfg> --split train --cand-stats <통계> --seed <s>`, `pr rerank --version v2 --config <cfg> --split val --cand-stats <통계>`.

## 6. 검증
- 입력 벡터 손계산(작은 고정 벡터), 정답 필드 미사용, 결정성(같은 seed → 같은 가중치·버전), val 미사용(학습 함수가 val 순위를 받지 않음), 실패 조건, v0·v1 동작 불변.
