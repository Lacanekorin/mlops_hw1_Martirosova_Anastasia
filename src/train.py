import os
import json
import yaml
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import joblib

import mlflow
import mlflow.sklearn

def main():
    #Загружаем параметры обучения из params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    train_params = params["train"]

    C = train_params["C"]
    max_iter = train_params["max_iter"]
    random_state = train_params["random_state"]

    #Загружаем подготовленные данные
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    #Отделяем параметры (X) и целевую переменную (y)
    target_col = "target"

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    #Создаем ML-пайплайн: стандартизация + логистическая регрессия
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            multi_class="auto"
        ))
    ])

    #Настраиваем подключение к MLflow-серверу

    mlflow.set_tracking_uri("http://localhost:5000")

    #Выбираем/создаем эксперимент с именем 'wine_classification_experiment'
    mlflow.set_experiment("wine_classification_experiment")

    #Запускаем новый MLflow run
    with mlflow.start_run(run_name="logreg_wine"):

        #Логируем параметры модели
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)

        #Обучаем модель
        model.fit(X_train, y_train)

        #Предсказываем на тестовых данных
        y_pred = model.predict(X_test)

        #Считаем accuracy
        acc = accuracy_score(y_test, y_pred)

        #Логируем метрику в MLflow
        mlflow.log_metric("accuracy", acc)

        #Сохраняем модель в файл model.pkl
        model_path = "model.pkl"
        joblib.dump(model, model_path)

        #Логируем файл модели как артефакт в MLflow
        mlflow.log_artifact(model_path)

        #Сохраняем метрики в metrics.json для DVC
        metrics = {"accuracy": acc}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)

        print(f"Accuracy на тесте: {acc:.4f}")
        print(f"Модель сохранена в {model_path}")
        print("Метрики сохранены в metrics.json")


if __name__ == "__main__":

    main()
