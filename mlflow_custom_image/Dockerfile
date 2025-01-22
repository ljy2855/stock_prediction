# Python slim 이미지를 기반으로 설정
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# Python 패키지 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# MLflow 서버를 기본 엔트리포인트로 설정
ENTRYPOINT ["mlflow"]
