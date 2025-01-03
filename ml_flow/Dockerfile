# MLflow 공식 이미지 사용
FROM ghcr.io/mlflow/mlflow:latest

# 사용자 및 그룹 생성
RUN groupadd -g 1001 mlflowgroup && \
    useradd -r -u 1001 -g mlflowgroup mlflowuser

# 디렉터리 생성 및 권한 설정
RUN mkdir -p /mlflow/mlruns /mlflow/db && \
    chown -R mlflowuser:mlflowgroup /mlflow/mlruns /mlflow/db

# SQLite DB 파일 생성 및 권한 설정
RUN touch /mlflow/db/mlflow.db && \
    chown mlflowuser:mlflowgroup /mlflow/db/mlflow.db

# 사용자 변경
USER mlflowuser

# 작업 디렉터리 설정
WORKDIR /mlflow

# MLflow 서버 실행
ENTRYPOINT ["mlflow", "server"]
CMD ["--backend-store-uri", "sqlite:///mlflow/db/mlflow.db", \
    "--default-artifact-root", "/mlflow/mlruns", \
    "--host", "0.0.0.0", \
    "--port", "5000"]
