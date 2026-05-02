"""실시간 매칭 푸시 패키지 — Phase 4-A.

governance.py       : opt-in 체크 · 위치 50m 양자화 · k-anonymity · push 한도
icp_embedder.py     : 전시자 ICP 자유 텍스트 → 구조화 + 임베딩
visitor_profiler.py : 참가자 행동/관심사 → 임베딩 (consent_matching=true 만)
matcher.py          : ICP↔Visitor cosine NN + 거리·시간 가중 점수
push_dispatcher.py  : FCM/APNs wrapper + 한도 enforce + audit log
"""
