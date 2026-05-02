"""ML 리드 스코어링 패키지 — Phase 3.

학습:    train.py    (XGBoost on training data exported from BigQuery)
추론:    predict.py  (Lambda inference + SHAP 설명, 룰베이스 폴백 안전망)
피처:    features.py (VisitorBehavior → numeric vector, 익명화)
모니터: monitor.py   (drift 감지, 200건 미만 시 룰베이스 강제)
"""
