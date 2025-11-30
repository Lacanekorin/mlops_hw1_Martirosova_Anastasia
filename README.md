# **MLOps Homework №1**
**Автор: Мартиросова Анастасия Гургеновна**

**Модуль 1: «Основы воспроизводимого машинного обучения. Жизненный цикл MLOps».**

# **1. Цель проекта**

Цель — построить полностью воспроизводимый ML-проект, 
в котором модель обучается для решения задачи классификации вина по химическим показателям, 
включающий:

* Git — контроль версий кода
* DVC — контроль версий данных и пайплайна
* MLflow — логирование параметров, метрик и артефактов

Используется датасет **Wine** из библиотеки `sklearn`.

# **2. Как запустить проект**

1. Клонирование репозитория

```
git clone https://github.com/Lacanekorin/mlops_hw1_Martirosova_Anastasia.git
cd mlops_hw1_Martirosova_Anastasia
```
2. Настройка окружения

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3. (Опционально) для macOS предварительно нужно указать переменную окружения

```
export CUSTOM_LOCALHOST='127.0.0.1'
```
4. Запустить MLflow UI

```
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

После запуска перейти в браузере:

**[http://localhost:5000](http://localhost:5000)**

(для macOS адрес: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

В MLflow доступны:

* параметры модели (`C`, `max_iter`)
* метрика `accuracy`
* артефакт `model.pkl`
* история экспериментов

5. Загрузка данных и воспроизведение всего пайплайна

```
dvc repro
```

# **3. Краткое описание пайплайна**

## **Подготовка данных**

Файл: `src/prepare.py`

* читает `data/raw/wine.csv`
* разделяет данные на **train** и **test**
* использует параметры из `params.yaml`
* сохраняет:

```
data/processed/train.csv
data/processed/test.csv
```

## **Обучение**

Файл: `src/train.py`

* читает подготовленные данные
* обучает `LogisticRegression` в пайплайне `StandardScaler + LogisticRegression`
* считает accuracy на тесте
* сохраняет:

```
model.pkl
metrics.json
```

* логирует параметры, метрики и артефакты в **MLflow**


