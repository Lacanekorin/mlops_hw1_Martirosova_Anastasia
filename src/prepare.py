import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    #Открываем файл params.yaml и читаем его содержимое как словарь
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    #Берем параметры именно для шага prepare
    prep_params = params["prepare"]
    test_size = prep_params["test_size"]
    random_state = prep_params["random_state"]

    #Загружаем сырой датасет wine из data/raw/wine.csv
    raw_path = os.path.join("data", "raw", "wine.csv")
    df = pd.read_csv(raw_path)

    #Делим данные на train и test
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"]
    )

    #Готовим папку data/processed для сохранения обработанных данных
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    #Пути к train и test файлам
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    #Сохраняем таблицы в CSV без индекса
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Обработанные данные сохранены:")
    print("  ", train_path)
    print("  ", test_path)


if __name__ == "__main__":

    main()
