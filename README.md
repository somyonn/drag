# Domain-Profile RAG (Capstone)

단일 노드에서 동작하는 **검색 증강 생성(RAG)** 베이스라인과, 동일한 통합 인덱스 위에서 **3개의 도메인 정책 프로파일**(`low_latency` / `privacy` / `freshness_accuracy`)을 비교 실험하는 프로젝트입니다.

- **코퍼스**: AWS · Docker · Google Drive 공식 문서를 수집해 단일 통합 인덱스(`data/index/official`)로 사용
- **임베딩/검색**: TF-IDF(`ngram 1-2`, max_features=5000) + FAISS(없으면 numpy 폴백)
- **생성**: OpenAI 호환 Chat Completions API, 키가 없거나 호출 실패 시 **Mock LLM** 자동 폴백
- **인터페이스**: CLI + FastAPI 기반 Web UI
- **계측**: 단계별 지연 시간(`latency_ms`)과 검색 점수를 `runs/*.jsonl`에 기록

## 파이프라인 개요

1. `data/docs/`의 텍스트 문서 로드/정규화 (`rag/indexing/ingest.py`)
2. 고정 크기 청킹 (size=500, overlap=100, `rag/indexing/chunk.py`)
3. TF-IDF 임베딩 (`rag/indexing/embed.py`)
4. 벡터 인덱스 빌드/저장 — FAISS 우선, numpy 폴백 (`rag/indexing/index.py`)
5. 질의 top-k 검색 (`rag/indexing/retrieve.py`) + 인덱스 인메모리 캐시 (`rag/indexing/cache.py`)
6. 프롬프트 구성 후 LLM 생성 (`rag/llm/generate.py`)
7. 지연/검색 지표 로깅 (`rag/core/timing.py`, `runs/*.jsonl`)

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LLM을 쓰려면 프로젝트 루트에 `.env`를 만듭니다(없으면 자동으로 Mock LLM 사용):

```bash
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
```

## 인덱스 빌드

```bash
python run_ingest.py --docs-dir data/docs --index-dir data/index/official
```

> 저장소에 이미 빌드된 `data/index/official`가 포함돼 있으면 이 단계는 생략 가능합니다.
> 단, 인덱스를 빌드한 scikit-learn 버전과 실행 환경 버전이 다르면 `InconsistentVersionWarning`이 뜨므로, 경고가 보이면 인덱스를 재생성하세요.

## 1) 베이스라인 단일 질의 (CLI)

```bash
python run_query.py --index-dir data/index/official --query "How do I use Docker Compose?"
```

## 2) 도메인 정책 프로파일 (CLI)

도메인별 정책은 `data/config/domain_profiles.json`에 정의되어 있습니다.

```bash
python scripts/run_domain_demo.py --profile low_latency --query "EC2 instance에서 IAM role 붙이는 방법은?"
python scripts/run_domain_demo.py --profile privacy --query "로그에서 이메일 주소는 어떻게 다뤄야 해?"
python scripts/run_domain_demo.py --profile freshness_accuracy --query "What is the current best practice for temporary AWS credentials?"
```

| 프로파일 | 핵심 정책 | retrieval_k / top_k |
|----------|-----------|---------------------|
| `low_latency` | 짧은 컨텍스트, 빠른 경로 우선 | 6 / 3 |
| `privacy` | 입력 질문·컨텍스트·답변 PII 마스킹(이메일/전화/AWS키/주민번호/신용카드/IP) + 유형별 마스킹 요약 | 5 / 3 |
| `freshness_accuracy` | 본문 날짜 우선(없으면 파일 mtime) freshness 재랭크, 높은 recall | 10 / 4 |

## 3) Web UI

```bash
python run_web.py
```

브라우저에서 http://127.0.0.1:8080 접속. 사이드바에서 프로파일/베이스라인, retrieval_k, top_k를 조절하고 단계별 지연 시간을 확인할 수 있습니다. LLM은 항상 클라우드 우선·Mock 폴백, KB는 항상 `data/index/official`입니다.

## 평가 (Evaluation)

### 베이스라인 지연 요약

```bash
python run_eval.py --queries data/queries/kb_eval.txt
```

### 멀티 모드 품질 + 지연 평가

베이스라인 + 3개 프로파일을 한 번에 돌려 지연 시간·검색 점수·코퍼스 적중을 리포트로 산출합니다. `--llm` 으로 실험 조건을, `--trials` 로 반복 측정을 제어합니다.

```bash
# 생성 시간 분포 + 검색 랭킹 정확도(실 LLM, mock 미사용) — API 키 필요
python scripts/run_kb_eval.py --llm cloud --trials 5 --concurrency 1   # -> runs/kb_eval_report_cloud.json
# 검색 지표만 재현성 확인(LLM 무관, 비용 0)
python scripts/run_kb_eval.py --llm mock  --trials 5 --concurrency 1   # -> runs/kb_eval_report_mock.json
# 처리량(throughput) 데모 — 병렬
python scripts/run_kb_eval.py --llm cloud --concurrency 8
```

- **`--llm {mock,cloud,auto}`**: `mock`=결정적 Mock LLM(네트워크 없음, 재현성), `cloud`=OpenAI API 전용(폴백 없음, **본 보고서 기본**), `auto`=cloud 우선·mock 폴백. 리포트에 `llm_condition`/`model_id`/`mock_answer_rate`가 기록됩니다.
- **`--trials N`**: 질의당 N회 반복 측정 → **생성 시간(generation)** 을 포함한 지연 분포(p50/p90/p95/p99·mean·std·min·max) 산출. 분산은 동시 부하를 배제하려 `--concurrency 1`로 측정하세요.
- 리포트 경로는 조건별로 자동 분리됩니다(`runs/kb_eval_report_{mock,cloud}.json`).

**정확도(Accuracy)는 두 층위로 측정**합니다:
- **(A) 검색 랭킹**(결정적, LLM 무관): `data/queries/kb_eval_labeled.jsonl`의 큐레이션된 정답 문서 집합(`relevant_uri_substrings`, OR-of-AND)으로 **MRR / nDCG@k / recall@k** 를 산출합니다. 코퍼스 라우팅(corpus accuracy)·느슨한 hit@k도 함께 계산됩니다.
- **(C) 생성 답변**(LLM-as-judge): 아래 `run_judge_eval.py` 참조.

기록되는 주요 지표:

- **latency_ms 분포**: 단계별(`index_load`, `retrieval`, `generation`, `total`)의 mean/std/p50/p90/p95/p99/min/max — 실 LLM에서 generation이 지배적
- **retrieval score / top_corpus**: top-k 평균 유사도와 소스 분포
- **검색 랭킹 정확도(라벨 있을 때)**: corpus accuracy, hit@k, MRR, nDCG@k, recall@k, keyword recall

### 답변 정확도 (LLM-as-judge)

생성 답변을 질의·검색 컨텍스트에 비추어 **관련성/충실도/정답성**(1~5점, RAGAS식, gold answer 불필요)으로 채점합니다. 생성 모델보다 강한 심판 모델을 써서 self-judge 편향을 완화합니다.

```bash
python scripts/run_judge_eval.py --llm cloud --judge-model gpt-4.1   # -> runs/judge_eval_report.json
```

- **`--judge-model`**(기본 `gpt-4.1`, env `OPENAI_JUDGE_MODEL`): 심판 모델. 생성(`OPENAI_MODEL`)과 다르게 두는 것을 권장합니다.
- 채점 축: `answer_relevance`(질문 적합) / `faithfulness`(컨텍스트 근거·환각 여부) / `correctness`(사실 정확). pass = 점수 ≥ 4.
- 모드별 평균·pass rate와 최저 점수 사례(`lowest_examples`)를 리포트에 기록합니다. 심판 출력은 JSON 강제(`response_format`) + 파싱 재시도로 안정화합니다.

### 프라이버시 마스킹 평가

합성 PII 코퍼스(`data/docs_privacy_test/`)에는 일반 PII와 함께 **난독화 PII(hard positive)** 및 **PII로 오인되기 쉬운 비PII(decoy)** 가 포함됩니다. 정답 라벨은 `data/queries/privacy_pii_truth.jsonl`.

```bash
python scripts/run_privacy_eval.py   # 전용 인덱스 자동 빌드, 결과: runs/privacy_eval_report.json
```

- **E2E 무결성**: 입력 질문 마스킹 재현율 + 마스킹된 컨텍스트 잔류 PII(누수) 측정
- **탐지기 PR 곡선**: conservative/balanced/aggressive 3개 operating point의 precision/recall/F1(유형별·micro·macro) + FN/FP 사례. 포맷 기반 탐지의 정밀도-재현율 상충을 드러냅니다.

### 최신성 정렬 평가

```bash
python scripts/run_freshness_eval.py # 전용 인덱스 자동 빌드, 결과: runs/freshness_eval_report.json
```

- **통제 정렬(`data/docs_freshness_test/`)**: 유사도를 균등화해 날짜만으로 순위가 정해지는 메커니즘 검증(top1/순서/Kendall tau)
- **trade-off ablation(`data/docs_freshness_hard/`)**: 가중치 w를 sweep해 **freshness 정렬(tau)** 과 **relevance@k**의 상충을 측정. 재랭크 점수는 `score + w·freshness`이며 `rerank_with_freshness(weight=...)` / 프로파일의 `freshness_weight` 로 조절됩니다.
- **mtime 폴백 열화(`data/docs_freshness_nodate/`)**: 본문 날짜가 없을 때 mtime 폴백으로 freshness 신호가 사라짐을 정량화

## 외부 문서 수집/동기화

웹/질의 경로에서 외부 HTTP 동기화는 **기본 비활성**입니다. 코퍼스를 갱신하려면 수동 스크립트를 사용합니다.

```bash
# 공식 문서 수집 (AWS · Docker · Google Drive)
python scripts/fetch_official_docs.py --sources all --max-pages 120

# 수동 외부 동기화 (allow_network 필요)
python scripts/sync_external_docs.py
```

## 프로젝트 구조

```
rag/                 # RAG 핵심 패키지 (기능별 서브패키지로 구성)
  pipeline.py        # 베이스라인 ingest/query 오케스트레이션
  core/              # 공통 기반: schemas, metrics, timing(단계별 지연 페이로드)
  indexing/          # 색인·검색: ingest, chunk, embed, index, cache(인메모리), lock, retrieve
  llm/               # LLM 클라이언트: generate (OpenAI / Mock / CloudThenMock)
  profiles/          # 정책 프로파일 질의 로직: query.py (저지연/정보보호/최신성)
  corpus/            # 코퍼스 수집·동기화: aws_fetch, doc_crawl, doc_sources,
                     #   external_sync, incremental_index, drift
web/                 # FastAPI 서버 + 정적 프런트엔드
scripts/             # 실행 진입점 (데모/평가/수집)
  _eval_common.py    # 평가 스크립트 공통 헬퍼(쿼리·라벨 로딩, 리포트 출력, 스레드 안전 로깅)
  run_kb_eval.py     # 지연 분포 + 검색 랭킹 정확도(MRR/nDCG/recall@k)
  run_judge_eval.py  # LLM-as-judge 답변 정확도 평가
  run_privacy_eval.py / run_freshness_eval.py  # 프라이버시·최신성 전용 평가
data/docs/           # 소스 문서 (aws / docker / google_drive)
data/docs_privacy_test/   # 합성 PII 평가 코퍼스 (일반 + 난독화 + decoy)
data/docs_freshness_test/ # 통제 최신성 코퍼스 (날짜만 다름)
data/docs_freshness_hard/ # 현실적 최신성 코퍼스 (관련성·최신성 상충)
data/docs_freshness_nodate/ # 날짜 없는 코퍼스 (mtime 폴백 시연)
data/index/official/ # 통합 TF-IDF 인덱스 + 메타
data/index/privacy_test/  # 프라이버시 평가 전용 인덱스
data/index/freshness_test/  # 통제 최신성 전용 인덱스
data/index/freshness_hard/  # 현실적 최신성 전용 인덱스
data/config/         # domain_profiles.json
data/queries/        # 평가용 질의셋·라벨 (kb / privacy_pii_truth / freshness_hard)
runs/                # 평가 리포트(*_report.json). 실행 로그(*.jsonl)는 gitignore
tests/               # 단위/통합 스모크 테스트
```

## 한계 및 향후 과제

- **프로파일 차별화**: 세 프로파일이 동일 인덱스를 공유하고 `top_k`가 비슷해, `low_latency`와 `baseline`의 지연 차이는 작습니다. 프로파일별 서브 인덱스나 retrieval_k 격차 확대가 향후 과제입니다.
- **임베딩/검색 정밀도**: 현재 TF-IDF 기반으로, 주제 영역은 잘 맞히지만(hit@k 0.9) 정답 문서를 상위에 정조준하는 정밀도는 낮습니다(MRR/nDCG ≈ 0.3~0.4). 신경망 임베딩(sentence-transformers 등) 도입 시 개선 여지가 크며, 검색 정밀도 향상은 생성 충실도(faithfulness) 개선으로 이어집니다.
- **정확도 평가의 한계**: 검색 랭킹은 큐레이션한 gold set(질의 10개)에 의존하고, LLM-as-judge는 심판 LLM의 변동·편향이 잔존합니다(다른 강한 모델·temperature=0로 완화). gold answer 기반 exact-match/F1, 다중 심판 self-consistency가 향후 과제입니다.
- **분산/엣지 구성**: 다중 엣지 노드, 라우터, drift-aware 재색인 등 분산 아키텍처는 본 구현 범위에 포함되지 않았으며 향후 확장 과제입니다.
