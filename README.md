# 📊 ML 기반 주가 예측 프로젝트

주가 데이터, 금리, 물가 상승률 등 경제 지표를 활용한 머신러닝 모델로 주가를 예측하고, 학습된 모델을 배포 및 관리

## 🎯 핵심 목표

1. 정확한 주가 예측 모델 구축
2. 실시간 데이터 파이프라인 운영
3. 모델 배포 및 관리 자동화
4. 사용자 맞춤형 투자 에이전트 제공

## 🛠️ 서비스

📌 1. Model Train & Serving

* 모델 학습: 주가 데이터와 경제 지표를 활용한 머신러닝 모델 학습.
* 모델 서빙: REST API를 통해 모델 예측값 제공.
* 버전 관리: MLflow를 사용한 모델 및 데이터셋 버전 관리.

📌 2. Web Service

* 개인화 에이전트: 사용자별 투자 성향에 맞춘 AI 에이전트 생성.
* 포트폴리오 관리: 에이전트를 통해 최적화된 포트폴리오 제공.
* 대시보드: 예측 결과, 투자 전략, 리스크 분석 시각화.

📌 3. Data Pipeline

* 실시간 경제 데이터 수집: 주가, 금리, 물가 상승률, 뉴스 데이터 수집.
* 데이터 전처리: 학습에 최적화된 데이터셋 구성.
* 자동화 파이프라인: ETL (Extract, Transform, Load) 파이프라인 구축.

### 폴더 구조

```bash
📂 stock_ml_project/
├── airflow/               # 데이터 파이프라인 관리
├── dashboard/            # 대시보드 및 사용자 관리 (Django)
├── docker-compose.yml    # 각 서비스 배포 (docker-compose)
├── grafana/              # Grafana 시각화 설정
├── ml_flow/              # MLflow 모델 학습 및 서빙
│   ├── config/          # 프로젝트 설정 파일
│   ├── data/            # 학습 및 테스트 데이터
│   ├── models/          # 학습된 모델 저장
│   ├── report/          # 모델 평가 및 백테스트 보고서
│   └── src/             # ML 모델 학습 및 평가 스크립트
└── model_serve/         # 모델 서빙 서버 (fastapi)
```

## 🔗 레퍼런스

### 머신러닝

[Time-Series Forecasting Transformer Models](https://arxiv.org/pdf/2304.04912)

[Financial Time Series Prediction](https://arxiv.org/pdf/2312.15235)

