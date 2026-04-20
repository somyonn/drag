# Basic RAG Baseline

This project provides a minimal, reproducible baseline RAG pipeline:

1. Load and normalize text documents from `data/docs/`
2. Chunk documents with fixed size/overlap
3. Embed chunks with TF-IDF
4. Build and persist a vector index (`faiss` if available, fallback to numpy)
5. Retrieve top-k chunks for a query
6. Generate an answer (mock LLM by default)
7. Log metrics and traces to `runs/logs.jsonl`

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
.env 파일 생성 후 OPENAI_API_KEY 설정
python run_ingest.py
python run_query.py --query "What is this project about?"
python run_eval.py --queries data/queries/sample_queries.txt
```

Example `.env`:

```bash
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
```

## Project structure

- `rag/`: reusable baseline RAG modules
- `data/docs/`: source documents
- `data/index/`: persisted index + metadata
- `runs/logs.jsonl`: execution traces and metrics

# Edge-Cloud Distributed RAG (Capstone)

단일 맥북에서 **논리적 엣지 3개 + 클라우드 공통 LLM** 구조를 시뮬레이션하는 **분산 RAG** 구현입니다.

## 목표
- **중앙집중형 RAG** vs **3-엣지 분산 RAG** vs **Drift-aware 분산 RAG** 비교
- 엣지별 도메인/정책(보안, 최신성, 우선순위, drift 임계치) 분리
- 클라우드는 **LLM 호출만 담당**, 검색은 엣지에서 수행

## 빠른 실행

### 1) 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) LLM 설정
기본은 **OPENAI 호환 Chat Completions API**를 사용합니다. `OPENAI_API_KEY`가 없으면 자동으로 **Mock LLM(오프라인)** 으로 폴백됩니다(파이프라인/실험 재현용).

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="gpt-4.1-mini"
# 필요 시 (OpenAI 호환 서버/프록시)
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### 3) 문서 인덱싱 + 질의 1회 실행
```bash
python -m rag_dist.cli --mode distributed --query "보안 로그에서 개인정보가 포함될 수 있는 필드는?"
```

기본 임베딩은 **오프라인 해시 임베딩**(즉시 실행/재현성)입니다. 신경망 임베딩을 쓰고 싶으면:
```bash
export USE_SENTENCE_TRANSFORMERS=1
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

### 4) 실행 모드
- `--mode centralized`: 중앙집중형 RAG(단일 인덱스)
- `--mode distributed`: 3-엣지 분산 RAG(기본: 라우터로 엣지 선택, `--user` 지정 시 사용자 전용 엣지로 고정)
- `--mode drift`: drift-aware 분산 RAG(드리프트 점검 후 필요 시 재인덱싱, `--user` 지정 시 사용자 전용 엣지로 고정)
- `--mode benchmark`: 위 3가지 구조를 동일 쿼리셋으로 비교 실행

#### 3명의 사용자(전용 RAG) 시뮬레이션
이 프로젝트에서 `user1/user2/user3`는 각각 **자기 전용 엣지 RAG**를 갖는 사용자로 동작합니다.

```bash
python -m rag_dist.cli --mode distributed --user user1 --query "p95 지연이 늘었을 때 점검은?"
python -m rag_dist.cli --mode distributed --user user2 --query "로그에서 이메일은 어떻게 처리해?"
python -m rag_dist.cli --mode distributed --user user3 --query "RAG 구성요소를 요약해줘"
```

`data/queries.jsonl`에도 `user_id` 필드를 넣을 수 있으며, `benchmark` 모드에서 해당 사용자 전용 엣지로 실행됩니다.

### 4) 비교 실험 실행(3가지 구조)
```bash
python -m rag_dist.cli --mode benchmark --queries data/queries.jsonl --out results.json
```

## 실험 지표(현재 JSON에 기록되는 항목)
- **latency_ms**: 전체 응답 지연 시간
- **ttft_ms**: 스트리밍이 아니라서 현재는 “LLM 호출/생성 시간”을 대체 지표로 기록
- **top1_score / topk_mean_score**: 검색 품질(유사도 기반 간이 지표)
- **reindex_count / reindex_triggered / reindex_reason**: 재색인 빈도 및 원인(드리프트 반응)

## 폴더 구조
- `rag_dist/`: 파이썬 패키지(엣지/클라우드/실험/유틸)
- `data/`: 엣지별 샘플 문서 + 질의셋

## 단계별 파일 추가(요약)
- **단계 A (MVP)**: `rag_dist/` 핵심 파이프라인 + `data/` 샘플 + `cli.py`
- **단계 B (Drift)**: `rag_dist/edge/drift.py` + 메트릭/재색인 트리거
- **단계 C (비교 실험)**: `rag_dist/experiments/*` + 지표 저장

## 단계별로 추가/핵심인 파일
- **단계 A (최소 실행)**\n  - `rag_dist/edge/edge_node.py`, `rag_dist/edge/router.py`\n  - `rag_dist/vectorstores/chroma_store.py`\n  - `rag_dist/cloud/llm_api.py`\n  - `rag_dist/cli.py`\n- **단계 B (Drift 추가)**\n  - `rag_dist/edge/drift.py`\n  - `rag_dist/experiments/drift_aware.py`\n- **단계 C (비교 실험)**\n  - `rag_dist/experiments/centralized.py`\n  - `rag_dist/experiments/distributed.py`\n  - `rag_dist/cli.py`의 `benchmark` 모드

