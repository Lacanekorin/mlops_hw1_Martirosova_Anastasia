import os
import pandas as pd
from sklearn.datasets import load_wine


def main():
    #Cоздаем папку data/raw, если ее нет
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    #Загружаем датасет Wine из sklearn
    wine = load_wine(as_frame=True)
    df = wine.frame  # DataFrame с признаками и колонкой target

    #Путь к выходному CSV
    raw_path = os.path.join(raw_dir, "wine.csv")

    #Сохраняем датасет в CSV
    df.to_csv(raw_path, index=False)

    print(f"Сырые данные Wine сохранены в {raw_path}")


if __name__ == "__main__":
    main()
