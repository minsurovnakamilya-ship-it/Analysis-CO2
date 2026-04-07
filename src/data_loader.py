import os
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

def prepare_timeseries_data(
    filepath, target_col="co2", item_col="country", timestamp_col="year"
):
    """
    Загружает данные CO2 и конвертирует их в формат AutoGluon.
    """
    # Загрузка
    df = pd.read_csv(filepath)

    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    n_rows, n_cols = df.shape
    print(f"Датасет загружен! Строк: {n_rows:,}, Колонок: {n_cols}")

    df = df.dropna(subset=[target_col, item_col, timestamp_col])

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="%Y")

   
    extra_cols = ["gdp", "population", "primary_energy_consumption"]
    available_cols = [col for col in extra_cols if col in df.columns]

    cols_to_use = [item_col, timestamp_col, target_col] + available_cols
    ag_df = df[cols_to_use].copy()

    ag_df = ag_df.rename(
        columns={
            item_col: "item_id",
            timestamp_col: "timestamp",
            target_col: "target",
        }
    )

   
    def fill_missing(group):
        return group.interpolate(method='linear').ffill().bfill()

   
    ag_df = ag_df.groupby("item_id", group_keys=False).apply(fill_missing)
    
    # Если в какой-то стране нет данных по параметру, ставим 0
    ag_df = ag_df.fillna(0)

    ts_data = TimeSeriesDataFrame.from_data_frame(ag_df)

    print(f"Данные готовы. Рядов (стран): {ts_data.num_items}")
    return ts_data

if __name__ == "__main__":
    path = "data/processed/owid_co2_data.csv"
    try:
        data = prepare_timeseries_data(path)
        print(data.head())
    except Exception as e:
        print(f"Ошибка: {e}")
