import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_save_model(X_train, X_test, y_train, y_test, model_path):
    # RandomForest 모델 초기화 및 훈련
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 모델 예측
    y_pred = model.predict(X_test)

    # 모델 성능 평가
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 결과 출력
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    # 훈련된 모델 저장
    joblib.dump(model, model_path)

    return model
