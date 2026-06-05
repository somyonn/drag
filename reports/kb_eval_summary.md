# KB 평가 결과 요약 (보고서용)

- 질의셋: `data/queries/kb_eval.txt` (10문항), 정답 라벨: `data/queries/kb_eval_labeled.jsonl` (랭킹용 gold 문서 집합 `relevant_uri_substrings` 포함)
- 인덱스: `data/index/official` (TF-IDF, 37,330 chunks, FAISS)
- 모드: `baseline` + 3개 도메인 프로파일

### 실험 조건 분리 (mock vs cloud) 및 재현성

`scripts/run_kb_eval.py`는 `--llm {mock,cloud,auto}` 로 LLM 조건을 명시적으로 분리한다. 조건별로 별도 리포트를 생성하고(`runs/kb_eval_report_{mock,cloud}.json`), `llm_condition`, `model_id`, `mock_answer_rate`, `trials` 를 함께 기록한다.

| 조건 | 클라이언트 | 용도 | 유효 지표 |
|------|-----------|------|-----------|
| `mock` | 결정적 Mock LLM (네트워크 없음) | 재현 가능한 검색 측정 | retrieval/index_load 지연, corpus_accuracy, hit@k, MRR/nDCG/recall@k |
| `cloud` | OpenAI API (폴백 없음) | E2E·생성 시간·답변 정확도 | generation 지연 분포, keyword_recall, LLM-as-judge |
| `auto` | cloud 우선, 실패 시 mock | 운영 기본값 | (혼합, mock_answer_rate로 구분) |

> 본 보고서의 지연·정확도 수치는 모두 **`--llm cloud`(실 LLM, mock 미사용)** 로 측정했다. 검색 랭킹 지표는 LLM과 무관하므로 mock에서도 동일하게 산출된다.

분산 측정을 위해 질의당 **N회 반복(`--trials N`)** 을 지원하며, p50/p90/p95/p99·mean·std·min·max를 산출한다. 분산은 동시 부하 영향을 배제하기 위해 `--concurrency 1`(순차)로 측정한다.

```bash
python scripts/run_kb_eval.py --llm cloud --trials 5 --concurrency 1   # 생성 시간 분포 + 검색 랭킹 정확도(본 보고서)
python scripts/run_kb_eval.py --llm mock  --trials 5 --concurrency 1   # 검색 지표만 재현성 확인(LLM 무관)
python scripts/run_judge_eval.py --llm cloud --judge-model gpt-4.1     # LLM-as-judge 답변 정확도
```

## 1. 지연 시간 분포

### 1-a. 검색·오버헤드 분포 (mock, trials=5, concurrency=1, n=50/mode)

근거: `runs/kb_eval_report_mock.json`. LLM 비용이 제거되어 **검색 파이프라인 자체의 변동성**을 드러낸다 (단위 ms).

| 모드 | mean | std | p50 | p90 | p95 | p99 |
|------|------|-----|-----|-----|-----|-----|
| baseline | 8.71 | 0.71 | 8.63 | 9.05 | 9.16 | 13.11 |
| low_latency | 8.57 | 0.39 | 8.52 | 8.83 | 9.62 | 10.03 |
| privacy | 8.56 | 0.24 | 8.58 | 8.81 | 8.94 | 9.09 |
| freshness_accuracy | 10.23 | 1.12 | 10.04 | 11.64 | 12.55 | 13.43 |

해석: `freshness_accuracy`는 본문 날짜 파싱을 위해 후보 문서를 읽으므로 평균·분산이 가장 크다(mean 10.2ms, std 1.12). 나머지 프로파일은 8.5~8.7ms로 근접하며, p99 꼬리(특히 baseline 13.1ms)는 인덱스 캐시/OS 스케줄링 지터에서 비롯된다.

### 1-b. 생성 시간(Generation Time) 분포 (cloud, mock 미사용)

근거: `runs/kb_eval_report_cloud.json` (실 LLM `gpt-4.1-mini`, `--llm cloud --trials 5 --concurrency 1`, mode당 n=50, `mock_answer_rate=0.00`). E2E 지연의 **99% 이상이 LLM 생성** 구간이며(검색 ~14ms, 인덱스 로드 캐시 적중 ~0.3ms), 아래는 **generation 구간만**의 분포다 (단위 ms).

| 모드 | mean | std | p50 | p90 | p95 | p99 |
|------|------|-----|-----|-----|-----|-----|
| baseline | 2843 | 1264 | 2405 | 4183 | 4546 | 8515 |
| low_latency | 3690 | 3459 | 2830 | 4651 | 10197 | 24118 |
| privacy | 2725 | 1163 | 2404 | 4054 | 4550 | 6737 |
| freshness_accuracy | 2616 | 1020 | 2331 | 3948 | 4694 | 6755 |

해석:
- **생성 시간은 검색 비용(~14ms)보다 2~3개 자릿수 크다**(평균 2.6~3.7초). 즉 정책별 retrieval 최적화는 E2E 체감 지연에 거의 영향을 주지 못하고, 지연의 지배 요인은 LLM API 응답이다.
- **변동성이 크다**: std가 평균에 맞먹는 수준이고 우측 꼬리가 길다(p99가 p50의 3~8배). 특히 `low_latency`는 단일 API 지터 outlier로 p99가 24초까지 치솟아, 프로파일 간 평균 차이(2.6~3.7초)는 네트워크 변동에 묻힌다.
- 따라서 "프로파일이 생성 시간을 줄인다"고 주장할 수 없으며, **저지연 정책의 이득은 retrieval 단계(1-a)에서, 체감 지연의 분산은 LLM 구간에서** 나온다고 분리 해석해야 한다.

## 2. 정확도(Accuracy) 지표

정확도를 두 층위로 측정한다: **(A) 검색 랭킹 정확도**(결정적, API 불필요)와 **(C) 생성 답변 정확도**(LLM-as-judge). 근거: `runs/kb_eval_report_cloud.json`, `runs/judge_eval_report.json`.

### 2-a. 검색 랭킹 정확도 (MRR / nDCG@k / recall@k)

각 질의에 대해 코퍼스 내 **정답 문서 집합(gold relevant set)** 을 큐레이션해(`relevant_uri_substrings`, OR-of-AND 규칙) 문서 단위 랭킹 지표를 산출한다. 정답 집합 크기는 질의당 1~7개이며, recall@k의 분모는 코퍼스 전체에서 규칙에 맞는 고유 문서 수다.

| 모드 | corpus acc | hit@k | MRR | nDCG@k | recall@k | keyword recall |
|------|-----------|-------|-----|--------|----------|----------------|
| baseline | 1.00 | 0.90 | 0.400 | 0.393 | 0.433 | 0.970 |
| low_latency | 1.00 | 0.90 | 0.400 | 0.393 | 0.433 | 0.970 |
| privacy | 1.00 | 0.90 | 0.400 | 0.393 | 0.433 | 0.980 |
| freshness_accuracy | 1.00 | 0.90 | **0.308** | **0.330** | 0.433 | 0.950 |

- **corpus accuracy = 1.0 / hit@k = 0.9**: 코퍼스 라우팅은 완벽하고, 느슨한 출처 적중도 9/10. 누락 1건은 Q9(Drive `permissions.create`)로 TF-IDF가 `about-sdk`를 상위로 검색.
- **MRR·nDCG는 1.00이 아니다(0.31~0.40)**: 느슨한 hit@k와 달리, *정확한* 정답 문서가 상위에 오는지를 보면 TF-IDF의 어휘 매칭 한계가 드러난다. 예컨대 Q1(`AllocateHosts`)·Q3(S3 vs IAM)·Q10(S3 접근통제)은 정답 문서가 top-k 밖이라 0점이다.
- **freshness_accuracy의 MRR/nDCG 하락(0.400→0.308, 0.393→0.330)**: 최신성 재랭크가 순수 관련성 순위를 흔들어 정답 문서의 순위를 떨어뜨리는 **trade-off를 정량적으로 보여준다**(같은 코퍼스라 recall@k 집합은 동일하나 순서가 나빠짐).
- **keyword recall(0.95~0.98)**: 실 LLM 답변에 기대 핵심어가 거의 포함됨. mock과 달리 실제 생성문 기준이다.

### 2-b. 생성 답변 정확도 (LLM-as-judge)

근거: `runs/judge_eval_report.json` · 스크립트: `scripts/run_judge_eval.py`. 생성 모델(`gpt-4.1-mini`)과 **다른 더 강한 심판 모델(`gpt-4.1`, temperature=0)** 로 self-judge 편향을 완화한다. 각 답변을 질의·검색 컨텍스트에 비추어 1~5점으로 채점(RAGAS식, gold answer 불필요). 40건 전부 채점 성공, mock 없음. pass = 점수 ≥ 4.

| 모드 | relevance | faithfulness | correctness | rel pass | faith pass | corr pass |
|------|-----------|--------------|-------------|----------|-----------|-----------|
| baseline | 4.60 | 3.60 | 3.70 | 0.90 | 0.60 | 0.60 |
| low_latency | 4.60 | 3.60 | 3.80 | 0.90 | 0.50 | 0.60 |
| privacy | 4.70 | 3.80 | 3.90 | 1.00 | 0.50 | 0.60 |
| freshness_accuracy | 4.70 | 3.80 | **4.00** | 0.90 | 0.60 | **0.70** |

- **answer relevance가 가장 높다(4.6~4.7)**: 답변이 질문을 잘 겨냥함.
- **faithfulness/correctness는 3.6~4.0으로 더 낮다**: 검색이 정답 문서를 못 가져온 질의에서 LLM이 컨텍스트에 없는 일반론을 생성(환각)해 충실도가 깎인다. 대표 사례는 Q10(S3 로그 PII) — 컨텍스트에 근거가 없어 일반 best-practice를 나열해 faithfulness 1점.
- **freshness_accuracy의 correctness가 약간 높다(4.00, corr pass 0.70)**: 최신성 재랭크가 일부 질의에서 더 적절한 근거 문서를 상위로 올린 효과로 보이나, 표본(질의 10개)이 작아 강한 결론은 유보한다.
- 핵심: **모든 지표가 1.00이 아니라 4.6/3.6/3.7처럼 분산**되어, 검색 미흡 → 생성 충실도 저하라는 RAG의 실제 약점을 드러낸다.

> 비고 — citation rate: 과거 표에서 freshness=1.0이던 것은 강제 출처 부착 때문이었고, 현재는 이를 제거(검색 정렬 품질을 인용으로 가리지 않기 위함)했으므로 LLM 자발 포함분만 집계된다. 최신성 정렬 품질은 5절 전용 평가로 별도 측정한다.

## 3. 질문별 결과 (baseline, cloud)

| # | 주제 | hit@k | MRR | nDCG@k | recall@k | kw recall |
|---|------|-------|-----|--------|----------|-----------|
| 1 | EC2 AllocateHosts 파라미터 | O | 0.00 | 0.00 | 0.00 | 1.00 |
| 2 | EC2 IAM 역할 연결 | O | 0.00 | 0.00 | 0.00 | 1.00 |
| 3 | S3 vs IAM 정책 | O | 0.00 | 0.00 | 0.00 | 0.50 |
| 4 | Lambda 환경 변수 제한 | O | 1.00 | 1.00 | 1.00 | 1.00 |
| 5 | KMS CMK vs AWS 관리 키 | O | 0.50 | 0.63 | 1.00 | 1.00 |
| 6 | Docker Compose 워크플로 | O | 0.50 | 0.30 | 0.33 | 1.00 |
| 7 | Dockerfile 멀티 스테이지 | O | 1.00 | 1.00 | 1.00 | 1.00 |
| 8 | Drive API files.list 검색 | O | 1.00 | 1.00 | 1.00 | 1.00 |
| 9 | Drive permissions.create | **X** | 0.00 | 0.00 | 0.00 | 1.00 |
| 10 | S3 로그 + PII + IAM/S3 통제 | O | 0.00 | 0.00 | 0.00 | 1.00 |

느슨한 hit@k(O/X)와 정밀한 랭킹 지표가 갈리는 지점이 핵심이다: Q1·Q3·Q10은 hit@k는 O이지만(같은 도메인 문서가 top-k에 있음) *정확한* 정답 문서는 상위에 없어 MRR/nDCG/recall@k=0이다. 즉 TF-IDF가 주제 영역은 맞히되 특정 정답 문서를 정조준하지 못하는 한계를 정량화한다.

## 4. 프라이버시 마스킹 평가

근거: `runs/privacy_eval_report.json` · 스크립트: `scripts/run_privacy_eval.py`

데이터셋(`data/docs_privacy_test/`)은 일반 PII 문서뿐 아니라 **난독화된 PII(hard positive)** 와 **PII로 오인되기 쉬운 비PII 숫자(hard negative/decoy)** 를 포함한다(정답 라벨 `data/queries/privacy_pii_truth.jsonl`).

### 4-a. 파이프라인 무결성 (E2E)

| 항목 | 값 | 비고 |
|------|----|------|
| 입력 질문 마스킹 재현율 | 1.00 | 6개 유형 모두 마스킹 후 LLM 전달 |
| 컨텍스트 잔류 PII(누수) | email 2, ip 2 | 운영(balanced) 마스킹이 **놓친** 난독화 PII (4-b 참조) |

- 입력 질문 → 컨텍스트 → 답변 순으로 마스킹되며, 답변 말미에 `[마스킹 요약]` 으로 유형별 건수를 고지한다.
- 누수 측정은 마스킹된 컨텍스트를 가장 공격적인 탐지기로 재스캔해 산출한다. balanced가 난독화 이메일과 콜론열(IPv6 형태)을 놓쳐 일부 PII가 컨텍스트에 잔류함을 정직하게 드러낸다.

### 4-b. 탐지기 정밀도-재현율 (operating points)

동일 정규식 탐지기를 3개 강도(conservative/balanced/aggressive)로 운용해 trade-off 곡선을 만든다(count 기반 micro 지표).

| 강도 | precision | recall | F1 |
|------|-----------|--------|----|
| conservative | 1.000 | 0.800 | 0.889 |
| balanced(운영 기본) | 0.861 | 0.886 | 0.873 |
| aggressive | 0.875 | 1.000 | 0.933 |

- **conservative**: 엄격 포맷만 매칭 → 오탐 0(정밀도 1.0)이지만 난독화/하이픈 없는 PII를 놓쳐 재현율 0.80.
- **aggressive**: 난독화 이메일·13자리 주민번호·IPv6까지 매칭 → 재현율 1.0이지만 날짜·주문번호 등을 오탐(정밀도 0.875).
- 즉 포맷 기반 탐지에는 **정밀도-재현율 상충**이 존재하며, 운영 기본값 balanced는 그 사이의 절충점이다. 합성셋에서 전 구간 1.00이 아니라 강도별로 곡선을 그린다는 점이 핵심이다.

## 5. 최신성 정렬 평가

근거: `runs/freshness_eval_report.json` · 스크립트: `scripts/run_freshness_eval.py`. 재랭크 점수는 `score + w·normalized_freshness`이며, 날짜는 **본문 날짜 우선(없으면 mtime)** 으로 파싱한다.

### 5-a. 통제된 정렬 메커니즘 (relevance 동일)

본문이 동일하고 날짜만 다른 6종(`data/docs_freshness_test/`)으로 유사도를 균등화해 순위를 날짜가 결정하도록 통제: top1 정확도·정확 순서 일치율·Kendall tau **모두 1.00**. 강제 출처 부착을 제거했으므로 citation_rate=0.

### 5-b. 관련성-최신성 trade-off (가중치 ablation)

유사도가 분산되고 "오래됐지만 관련성 높은" 문서와 "최신이지만 관련성 낮은" distractor를 섞은 현실적 코퍼스(`data/docs_freshness_hard/`)에서 가중치 w를 sweep한 결과(top_k=4):

| w | relevance@k | freshness Kendall tau |
|---|-------------|-----------------------|
| 0.0 | 1.00 | 0.333 |
| 0.05 | 1.00 | 0.333 |
| 0.15 | 1.00 | 0.333 |
| 0.30 | 1.00 | 0.333 |
| 0.50 | 1.00 | **0.667** |
| 1.00 | **0.75** | -0.333 |

- 적정 가중치(w≈0.5)에서는 **관련성을 유지(relevance@k=1.0)** 하면서 최신성 정렬이 개선된다(tau 0.33→0.67).
- 가중치를 과도하게(w=1.0) 높이면 최신이지만 무관한 문서가 top-k에 진입해 **relevance@k가 0.75로 하락**하고, 오히려 정렬도 흐트러진다. 단순한 task가 아니라 관련성과 최신성의 명확한 상충이 드러난다.

### 5-c. mtime 폴백 열화

| 코퍼스 | 본문 날짜 | mtime 폴백 | 타임스탬프 span | freshness 신호 |
|--------|-----------|-----------|------------------|----------------|
| `docs_freshness_hard` (날짜 임베드) | 6/6 | 0 | ~5년 | 사용 가능 |
| `docs_freshness_nodate` (날짜 없음) | 0/3 | 3/3 | 0초 | **사용 불가** |

본문에 날짜가 없으면 mtime으로 폴백하는데, 크롤링 코퍼스는 mtime이 균일(span≈0)해 freshness 신호가 사실상 사라진다. 즉 최신성 전략의 효과는 **발행/수정일 메타데이터 확보 여부에 의존**한다.

## 6. 한계 및 향후 과제

- **프로파일 지연 차별화 미미**: 세 프로파일이 동일 인덱스를 공유하고 retrieval 비용이 ~15ms로 작아, total 지연 차이는 LLM 응답 변동(p95에서 freshness 10초 등)에 묻힌다. 프로파일별 서브 인덱스, retrieval_k 격차 확대, 또는 검색 단독 벤치마크로 분리 측정이 필요하다.
- **검색 정밀도 한계(MRR/nDCG 0.31~0.40)**: TF-IDF 어휘 매칭은 주제 영역은 맞히되(hit@k 0.9) 정답 문서를 상위에 정조준하지 못한다. 신경망 임베딩(sentence-transformers 등) 도입 시 의미 검색으로 MRR/nDCG 개선 여지가 크다. 또한 정답 문서가 top-k에 없으면 생성 충실도(2-b)가 동반 하락한다.
- **정확도 측정의 두 층위와 한계**: (A) 검색 랭킹은 큐레이션한 gold relevant set 기반으로 결정적이나 정답 집합 선정에 주관이 개입한다(질의 10개로 표본 작음). (C) LLM-as-judge는 생성 모델과 다른 강한 심판 모델(gpt-4.1)·temperature=0로 편향·분산을 완화했으나, 심판 LLM 자체의 변동·편향은 잔존한다. gold answer 기반 exact-match/F1과 다중 심판(self-consistency) 교차검증이 향후 과제. `keyword_recall`은 보조 근사치로 유지한다.
- **실험 조건 분리**: `--llm mock`(검색·품질, 재현 가능) vs `--llm cloud`(E2E 지연)을 명시적으로 분리하고 `llm_condition`/`model_id`/`mock_answer_rate`를 리포트에 기록한다. mock 조건에서는 generation 지연이 무의미하므로 retrieval/품질 지표만 해석한다.
- **마스킹은 포맷 기반(정밀도-재현율 상충)**: 정규식 탐지는 강도에 따라 conservative(P 1.0/R 0.80)~aggressive(P 0.875/R 1.0)의 곡선을 그리며, 운영 balanced는 난독화 PII 일부를 컨텍스트에 누수시킨다. 문맥 기반 NER 탐지 도입과 외부 실데이터에 대한 오탐·미탐 검증이 향후 과제.
- **최신성은 파싱 가능한 날짜에 의존**: 본문 날짜가 없으면 mtime 폴백인데 크롤링 코퍼스는 mtime이 균일(span≈0)해 신호가 사라진다. 또한 가중치가 과도하면 관련성이 희생되므로(5-b), 적정 가중치 선택과 발행/수정일 메타데이터 수집이 향후 과제.
- **합성 평가셋의 한계**: 통제·정답 확보를 위해 합성 코퍼스를 사용했다. 절대 수치보다 **조건/강도/가중치에 따른 추세(trade-off)** 가 핵심이며, 실데이터 일반화는 별도 검증이 필요하다.
