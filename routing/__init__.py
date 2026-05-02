"""동선 최적화 패키지 — Phase 4-C.

crowd_tracker.py  : BoothCrowdState 추적 (in-memory dict, 실 운영 Redis Streams)
graph_builder.py  : 부스 그래프 (networkx 또는 순수 python adjacency)
recommender.py    : 0.4·interest + 0.2/dist + 0.3/(1+crowd) + 0.1·persona
fastpass.py       : MyPass 챌린지 통합 — 패스트트랙 인센티브 부여
"""
