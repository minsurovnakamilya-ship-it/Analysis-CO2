import os

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame


def prepare_timeseries_data(
    filepath, target_col="co2", item_col="country", timestamp_col="year"
):
    """
    Загружает данные CO2 и конвертирует их в формат AutoGluon.
    """
    # 1. Загрузка
    df = pd.read_csv(filepath)

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    n_rows, n_cols = df.shape

    print(
        f"✅ Датасет загружен! "
        f"Строк: {n_rows:,} "
        f"Колонок: {n_cols} "
        f"Размер файла: {file_size_mb:.1f} MB"
    )

    # 2. Очистка: AutoGluon не любит пропуски в целевой переменной
    df = df.dropna(subset=[target_col, item_col, timestamp_col])

    # 3. Форматирование времени
    # Превращаем год (int) в объект datetime, чтобы библиотека понимала частоту данных
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="%Y")

    # 4. Выбор нужных колонок для модели
    # AutoGluon требует: item_id (сущность), timestamp (время), target (что предсказываем)
    ag_df = df[[item_col, timestamp_col, target_col]].copy()
    ag_df = ag_df.rename(
        columns={item_col: "item_id", timestamp_col: "timestamp", target_col: "target"}
    )

    # 5. Преобразование в специальный формат AutoGluon
    ts_data = TimeSeriesDataFrame.from_data_frame(ag_df)

    print(f"Данные загружены. Рядов (стран): {ts_data.num_items}")
    return ts_data


if __name__ == "__main__":
    # Тестовый запуск
    path = "data/processed/owid_co2_data.csv"
    try:
        data = prepare_timeseries_data(path)
        print(data.head())
    except FileNotFoundError:
        print("Файл не найден. Проверьте путь к data/raw/")
