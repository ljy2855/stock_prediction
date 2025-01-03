import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

# FastAPI 애플리케이션 초기화
app = FastAPI()

# MLflow Tracking URI 설정
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"

# 모델 캐시
model_cache = {}


class InferenceRequest(BaseModel):
    inputs: List[List[float]]


@app.get("/")
def read_root():
    return {"message": "MLflow FastAPI Model Serving Server is running 🚀"}


@app.get("/models/")
def list_models():
    """
    등록된 모델 목록 반환
    """
    import mlflow
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()
    return [{"name": model.name, "latest_version": model.latest_versions[0].version} for model in models]


@app.get("/models/{model_name}/versions/")
def list_model_versions(model_name: str):
    """
    특정 모델의 버전 목록 반환
    """
    import mlflow
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(model_name)
    return [{"version": version.version, "status": version.status} for version in versions]


@app.post("/models/predict/{model_name}/{version}/")
@app.post("/models/predict/{model_name}/")
def predict(model_name: str, version: Optional[str] = None, request: InferenceRequest = None):
    """
    특정 모델 버전을 사용하거나, 버전이 없을 경우 최신 버전으로 예측 수행
    """
    global model_cache
    client = mlflow.tracking.MlflowClient()

    # 버전이 제공되지 않은 경우 최신 버전 사용
    if version is None:
        try:
            versions = client.get_latest_versions(model_name)
            if not versions:
                raise HTTPException(status_code=404, detail=f"No versions found for model: {model_name}")
            version = versions[0].version  # 최신 버전 선택
            print(f"ℹ️ Using latest version ({version}) of model '{model_name}'")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch latest model version: {e}")

    # 모델 키 생성 (캐시 확인용)
    model_key = f"{model_name}:{version}"
    if model_key not in model_cache:
        try:
            model_uri = f"models:/{model_name}/{version}"
            print(f"ℹ️ Loading model from URI: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            model_cache[model_key] = model
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # 예측 수행
    try:
        predictions = model_cache[model_key].predict(request.inputs)
        return {"model_name": model_name, "version": version, "predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")