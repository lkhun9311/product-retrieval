# C5 재랭커 v1 계약 (이미지 단위 후보 특징)

초안: 2026-10-05 (메인). 이슈 #18. v0 계약(`c5-reranker-v0.md`)의 범위·학습·버전·추론 규칙을 그대로 따르고, 특징만 늘린다.
학습 곡선(#11)·비교 평가(#10)를 보기 전에 고정한다. 사전등록 기준(목표 0.887, 측정 지점)은 바꾸지 않는다.

## 1. 후보 통계 파일 (순위 파일과 분리)
- 순위 파일(`pr eval` 출력)은 바꾸지 않는다. c4-v3의 시뮬레이션 namespace가 순위 파일 sha256에 묶여 있어서다.
- `pr cand-stats --config <cfg> --split <train|val> --rankings <순위 파일>`이 같은 index·같은 질의 임베딩(캐시)으로
  `<순위 파일 stem>.cand_stats.jsonl`을 쓴다. 한 줄 = 한 질의: `query_id`, `rankings_sha256`, `index_id`, `stats`(순위 파일 상위 20과 같은 순서의 목록).
- 후보 한 개의 통계 (질의 벡터 q, 그 상품의 갤러리 이미지 벡터 g_1..g_n, 모두 L2 정규화, 유사도 = 내적):
  - `n_images` = n
  - `max_sim` = max(q·g) — 순위 파일 `scores`와 같아야 한다(1e-5 이내, 다르면 실패)
  - `mean_sim` = mean(q·g)
  - `second_sim` = 두 번째로 큰 q·g (n = 1이면 max_sim)
  - `std_sim` = q·g의 모표준편차 (n = 1이면 0)
  - `top1_sim` = 이 상품의 평균 벡터(정규화)와 1위 상품의 평균 벡터(정규화)의 내적 (1위 자신은 1.0)
- 정답 필드는 읽지 않는다. 질의 → 이미지 sha 대응은 매니페스트에서 읽는다.

## 2. 특징 (v1 = 10개)
v0 5개(`score`, `gap_top1`, `gap_next`, `zscore`, `log_rank`) + 
6. `log_n_images` = ln(1 + n_images)
7. `mean_sim`
8. `gap_second` = max_sim − second_sim
9. `std_sim`
10. `top1_sim`

## 3. 학습·추론·버전
- v0와 같다(상위 20, 표준화 + L2 로지스틱 회귀 C = 1.0, sample_weight = 라벨 weight, JSON 저장, numpy 추론).
- 계약 문자열 "c5-rerank-v1". 버전 해시에 후보 통계 파일 sha256을 더한다.
- 학습·추론 때 순위 파일과 통계 파일의 `rankings_sha256`·query_id·후보 순서가 맞지 않으면 실패한다.
- `pr train-rerank`·`pr rerank`에 `--version v0|v1`(기본 v0)과 `--cand-stats PATH`를 더한다. v0 동작은 바뀌지 않는다.

## 4. 검증
- 통계 손계산(작은 고정 벡터), `max_sim` = 순위 점수 대조, 순위 파일 sha256 불변, 정답 필드 미사용, 불일치 실패, v0 테스트 그대로 통과.
