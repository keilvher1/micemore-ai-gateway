# MICEMORE AI Gateway

부스 코파일럿(RAG), 라이브 통역, 리드 스코어링 등 LLM 호출을 모은 단일 게이트웨이.
Phase 1 — `/copilot/query` (SSE 스트리밍) 만 활성.

## 빠른 시작 (mock 모드)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .env 에서 USE_MOCK=true 인지 확인
uvicorn main:app --reload --port 8000
```

테스트:

```bash
curl -N -X POST http://localhost:8000/copilot/query \
  -H "Content-Type: application/json" \
  -d '{"booth_id":"lumen","session_id":"s1","question":"What does this product do?","target_lang":"en","mock":true}'
```

## 프로덕션 — Pinecone 인덱싱

```bash
python -m rag.ingest --booth lumen --pdf /tmp/lumen_catalog.pdf
```

## 배포 — AWS Lambda Container

```bash
docker build -t micemore-ai-gateway .
docker tag micemore-ai-gateway:latest <ecr-url>
docker push <ecr-url>
# Lambda 콘솔에서 Image URI 갱신 + API Gateway 연결
```

## 디렉토리

```
.
├── main.py              # FastAPI app, CORS, Sentry, lifespan
├── routes/copilot.py    # POST /copilot/query (SSE)
├── rag/
│   ├── pipeline.py      # LangChain RAG (Pinecone + Claude streaming)
│   ├── embeddings.py    # text-embedding-3-small wrapper
│   └── ingest.py        # PDF → chunk → embed → Pinecone upsert
├── prompts/copilot_system.py  # system prompt builder
├── models/trigger_event.py    # 다채널 입력 추상화
├── requirements.txt
├── Dockerfile           # Lambda Container
└── .env.example
```

## 테스트

```bash
USE_MOCK=true python3 -m pytest tests/ -v
```

검증된 14 tests:
- `test_copilot_route.py` (5) — health, SSE 순서, 한국어 응답, 검증 실패 케이스
- `test_prompt_builder.py` (6) — 다국어, 출처 주입, 600자 truncate, 빈 source
- `test_trigger_event.py` (3) — 6채널 동일 직렬화, 잘못된 lang, NFC 페이로드 round-trip

## Mock 모드

`USE_MOCK=true` 일 때:
- LLM / 임베딩 / Pinecone 호출 안 함
- 고정 응답을 토큰 단위로 흘려보냄 (40ms 간격)
- CI / 로컬 개발 / 데모 fallback 에서 사용

## 한국관광공사 TourAPI 4.0 활용 (2026 관광데이터 활용 공모전)

본 게이트웨이는 한국관광공사 OpenAPI 의 **9 개 오퍼레이션** 을 단순 호출이 아니라
**AI 학습 데이터** 로 인제스트하여 RAG 응답에 직접 활용합니다 — 다른 참가자와의 핵심 차별점.

### 9 개 엔드포인트 (`routes/tourapi.py`)

| 경로 | KTO 오퍼레이션 | 용도 |
|---|---|---|
| `GET /tourapi/festivals` | searchFestival2 | 다음 행사 매칭 (booth-match) |
| `GET /tourapi/nearby` | locationBasedList2 | 부스 주변 추천 (saved-booth-detail) |
| `GET /tourapi/match-bonus` | searchFestival2 | 사용자 관심 지역 향후 30 일 행사 |
| `GET /tourapi/area-list` | areaCode2 | 지역/시군구 메타 |
| `GET /tourapi/detail` | detailCommon2 | 행사 / POI 상세 (RAG 인제스트 원천) |
| `POST /tourapi/sync` | (composite) | 지역 + 행사 동기화 1-shot |
| `POST /tourapi/ingest` | + Pinecone | 인제스트 풀/증분 트리거 (CLI / Cron) |
| `GET  /tourapi/search-rag` | + Pinecone | tour 네임스페이스 직접 RAG 검색 |
| `POST /tourapi/cache/refresh` | (composite) | 4 권역 × 60 일 캐시 갱신 (D-1 cron) |

### 인제스트 CLI (`rag/kto_ingest.py`)

```bash
# 전체 인제스트 (포항 + 서울 + 경북 + 부산 — 8 행사 × 4 언어 × 5 청크 = 160 vectors)
python -m rag.kto_ingest --mode full --dry-run

# 증분 동기화 (특정 일자 이후만)
python -m rag.kto_ingest --mode sync --since 20260501

# Pinecone 실 upsert (KTO_TOURAPI_KEY + PINECONE_API_KEY 실값 필요)
python -m rag.kto_ingest --mode full
```

### Copilot 의 source 라우팅

`POST /copilot/query` 의 `source` 파라미터로 namespace 라우팅:

- `booth` (기본) — 부스 자료실만 (`booth-{booth_id}`)
- `tour` — 한국관광공사 RAG 만 (`tour:{lang}:{areacode}`)
- `auto` — 둘 다 검색 후 score 상위 k 청크 합산 (Boomi Chat 기본 모드)

`KTO_TOURAPI_KEY=PLACEHOLDER` 면 자동으로 mock fixture 폴백 — 키 발급 전에도 시연 100% 가능.
