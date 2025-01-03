import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from functools import wraps


def mlflow_log_training(func):
    """
    AOP 방식으로 학습 메서드에 MLflow 로깅 기능을 추가하는 데코레이터
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 기본 파라미터 로깅
        mlflow.log_param("model_name", self.model_name)
        mlflow.log_param("learning_rate", self.optimizer.param_groups[0]['lr'])
        mlflow.log_param("device", str(self.device))
        mlflow.log_param("num_epochs", kwargs.get("num_epochs", 50))
        mlflow.log_param("early_stopping", kwargs.get("early_stopping", False))
        mlflow.log_param("patience", kwargs.get("patience", 10))
        mlflow.log_param("total_params", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        
        # 모델 레이어 상세 정보 로깅
        layer_details = self.get_model_details()
        mlflow.log_dict({"layer_details": layer_details}, "model_layer_details.json")

        # 원래 학습 메서드 실행
        loss_history = func(self, *args, **kwargs)

        # 안전하게 샘플 입력 가져오기
        try:
            train_loader = args[0]  # train_loader는 첫 번째 위치 인자
            sample_batch = next(iter(train_loader))
            sample_input, _ = sample_batch
            sample_input = sample_input.to(self.device)
            model_output = self.model(sample_input).detach().cpu().numpy()
            signature = infer_signature(sample_input.cpu().numpy(), model_output)
        except Exception as e:
            raise ValueError(f"Failed to infer model signature: {e}")

        # 모델 저장
        mlflow.pytorch.log_model(
            pytorch_model=self.model,
            artifact_path="transformer-model",
            signature=signature,
            registered_model_name="pytorch-transformer-time-series-model"
        )

        print("✅ MLflow: Training parameters, loss history, layer details, and model logged successfully.")
        return loss_history
    return wrapper


def mlflow_log_evaluation(func):
    """
    AOP 방식으로 평가 메서드에 MLflow 로깅 기능을 추가하는 데코레이터
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        mse, r2, adjusted_r2 = func(self, *args, **kwargs)

        # 평가 메트릭 로깅
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("adjusted_r2_score", adjusted_r2)

        print("✅ MLflow: Evaluation metrics logged successfully.")
        return mse, r2, adjusted_r2
    return wrapper
