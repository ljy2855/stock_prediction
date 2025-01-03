import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
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


@app.post("/models/{model_name}/versions/{version}/predict/")
def predict(model_name: str, version: str, request: InferenceRequest):
    """
    특정 모델 버전을 사용하여 예측 수행
    """
    global model_cache

    model_key = f"{model_name}:{version}"
    if model_key not in model_cache:
        try:
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.pyfunc.load_model(model_uri)
            model_cache[model_key] = model
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    try:
        predictions = model_cache[model_key].predict(request.inputs)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
