import joblib
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model_path, X_test, y_test):
    # 저장된 모델 로드
    model = joblib.load(model_path)

    # 예측
    y_pred = model.predict(X_test)

    # 평가 지표 계산
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 결과 출력
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")
