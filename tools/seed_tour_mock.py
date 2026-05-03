"""seed_tour_mock.py — KTO 키 없이 mock seed 경북(35) 콘텐츠를 OpenAI 임베딩 + Pinecone 적재.

D-2 시연 보강용. 실 KTO 키 주입 후에는 `python -m rag.kto_ingest --mode full` 사용.

USAGE:
    python -m tools.seed_tour_mock

산출:
    micemore-tour 인덱스의 namespace=tour:ko:35 / tour:en:35 에 4 콘텐츠씩 upsert.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


# 4 콘텐츠 × 2 언어 = 8 문서 (시연용 풍부 description).
DOCS = [
    {
        "contentid": "3110308",
        "title_ko": "POEX 2026 — 포항국제전시회",
        "title_en": "POEX 2026 — Pohang International Exhibition",
        "ko": (
            "포항국제전시회 (POEX 2026) 는 동남권 최대 규모 산업 전시회로 "
            "첨단 소재·반도체·에너지 분야 250개 부스가 참가합니다. "
            "2026년 9월 10일부터 12일까지 3일간 포항 송도 컨벤션센터에서 열리며, "
            "외국인 바이어 미팅과 한국관광공사 K-Tourism Brand 동반 진출 프로그램이 포함됩니다. "
            "사전 등록은 무료, 현장 등록은 30,000원이고 학생은 50% 할인입니다. "
            "주차는 컨벤션센터 무료 200대 가능하고, KTX 포항역에서 셔틀버스가 운영됩니다."
        ),
        "en": (
            "POEX 2026 (Pohang International Exhibition) is the largest industrial expo in "
            "southeast Korea with 250 booths in advanced materials, semiconductors, and energy. "
            "Held Sep 10-12 2026 at Songdo Convention Center, Pohang. "
            "Includes overseas buyer meetings and Korea Tourism Org K-Tourism Brand showcase. "
            "Free pre-registration, 30,000 KRW on-site, 50% student discount. "
            "Free parking 200 spots, shuttle from KTX Pohang station."
        ),
        "areacode": "35",
        "contenttypeid": "15",
    },
    {
        "contentid": "126508",
        "title_ko": "송도해수욕장 — 포항",
        "title_en": "Songdo Beach — Pohang",
        "ko": (
            "포항 송도해수욕장은 포항 시내에서 가장 가까운 백사장 (길이 1.7km, 폭 70m) 으로 "
            "가족 단위 휴양객에게 인기있는 곳입니다. 모래가 곱고 수심이 완만해서 어린이 동반에 좋고, "
            "여름철에는 야간 개장과 무료 와이파이 서비스가 제공됩니다. "
            "도보 10분 거리에 죽도시장 회 거리가 있어서 점심·저녁 식사 후 산책 코스로 추천됩니다."
        ),
        "en": (
            "Songdo Beach in Pohang is the closest white-sand beach to downtown (1.7km long, 70m wide) "
            "and popular among families. Fine sand and gentle slope make it safe for children. "
            "Summer night opening with free WiFi. "
            "Jukdo Market sashimi alley is within a 10-minute walk — great post-meal stroll route."
        ),
        "areacode": "35",
        "contenttypeid": "12",
    },
    {
        "contentid": "129380",
        "title_ko": "영일대해수욕장 — 포항",
        "title_en": "Yeongildae Beach — Pohang",
        "ko": (
            "포항 영일대해수욕장은 영일만 해변 위로 떠 있는 영일대 정자가 상징인 야경 명소입니다. "
            "백사장 길이 1km, 가족 단위 휴식과 산책에 적합하며 인근에 카페 거리가 활성화되어 있습니다. "
            "부산 해운대보다 한적해서 여유로운 휴식을 원하는 방문객에게 좋습니다. "
            "POEX 2026 컨벤션센터에서 차로 15분 거리입니다."
        ),
        "en": (
            "Yeongildae Beach in Pohang features a pavilion floating over Yeongil Bay — a famous "
            "night-view spot. 1km of beach great for family relaxation and walks, lined with cafes. "
            "Quieter than Busan Haeundae, ideal for visitors seeking a relaxed break. "
            "15-minute drive from POEX 2026 convention center."
        ),
        "areacode": "35",
        "contenttypeid": "12",
    },
    {
        "contentid": "2840125",
        "title_ko": "경주 실크로드 컨퍼런스 2026",
        "title_en": "Gyeongju Silk Road Conference 2026",
        "ko": (
            "경주 실크로드 컨퍼런스 2026 은 동서양 문화 교류를 주제로 한 학술 컨퍼런스입니다. "
            "2026년 8월 20일부터 22일까지 경주 신평동 컨벤션센터에서 열리며, "
            "전 세계 역사·문화 학자 200여 명이 참가합니다. "
            "경주는 세계문화유산이 밀집된 천년 고도로 가족 동반 여행에도 적합합니다. "
            "불국사·석굴암·첨성대 등 주요 유적이 30분 이내 거리에 위치합니다."
        ),
        "en": (
            "Gyeongju Silk Road Conference 2026 is an academic conference on East-West cultural exchange. "
            "Held Aug 20-22 2026 at Sinpyeong Convention Center, Gyeongju, with 200+ history and culture "
            "scholars from around the world. "
            "Gyeongju is a 1000-year-old capital dense with UNESCO World Heritage sites — also great for "
            "family travel. Major sites Bulguksa, Seokguram, Cheomseongdae all within 30 minutes."
        ),
        "areacode": "35",
        "contenttypeid": "15",
    },
]


def main() -> int:
    if "PLACEHOLDER" in os.getenv("OPENAI_API_KEY", "PLACEHOLDER") or \
       "PLACEHOLDER" in os.getenv("PINECONE_API_KEY", "PLACEHOLDER"):
        print("[error] OPENAI_API_KEY or PINECONE_API_KEY is placeholder — abort.")
        return 2

    import openai  # type: ignore
    from pinecone import Pinecone  # type: ignore

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("micemore-tour")
    client = openai.OpenAI()
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    total = 0
    for lang in ("ko", "en"):
        ns = f"tour:{lang}:35"
        vectors = []
        for d in DOCS:
            text = d[lang]
            title = d[f"title_{lang}"]
            full = f"{title}\n\n{text}"
            emb = client.embeddings.create(model=embed_model, input=full).data[0].embedding
            vectors.append({
                "id": f"kto:{d['contentid']}:{lang}",
                "values": emb,
                "metadata": {
                    "contentid": d["contentid"],
                    "title": title,
                    "doc_title": title,
                    "page": 1,
                    "text": full,
                    "areacode": d["areacode"],
                    "contenttypeid": d["contenttypeid"],
                    "lang": lang,
                    "source": "kto_mock_seed",
                },
            })
        index.upsert(vectors=vectors, namespace=ns)
        print(f"[upsert] ns={ns} vectors={len(vectors)}")
        total += len(vectors)

    print(f"[done] total upserted = {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
