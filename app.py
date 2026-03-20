import os

import streamlit as st
import pandas as pd
from src.data_loader import prepare_timeseries_data
from src.model import ForecastModel
from src.visualize import plot_forecast

# Настройка страницы
st.set_page_config(page_title="CO2 Forecast System", layout="wide")
st.title("🌍 Система прогнозирования выбросов CO2")
st.markdown(
    "Интеграция **AutoGluon TimeSeries** для анализа глобальных данных (1750-2024)"
)

# 1. Загрузка данных

DATA_PATH = "data/processed/owid_co2_data.csv"


@st.cache_data  # Кэшируем, чтобы не читать файл при каждом клике
def load_data():
    return prepare_timeseries_data(DATA_PATH)


try:
    ts_data = load_data()
    # Список уникальных стран для выбора
    countries = ts_data.index.get_level_values(0).unique().tolist()

    # Боковая панель (Sidebar)
    st.sidebar.header("Настройки прогноза")
    selected_country = st.sidebar.selectbox(
        "Выберите страну/регион",
        countries,
        index=countries.index("World") if "World" in countries else 0,
    )
    prediction_years = st.sidebar.slider("Горизонт прогноза (лет)", 5, 30, 10)

    # 2. Инициализация модели
    model = ForecastModel(model_path="models/ag_model_v1")

    # Кнопка обучения (если модель еще не создана)
    if st.sidebar.button("Обучить модель (AutoGluon)", key="train_button"):
        with st.spinner("Обучение модели..."):
            model.train(ts_data, prediction_length=prediction_years)
        st.sidebar.success("Модель обучена!")

    # 3. Получение прогноза и Визуализация
    if st.button("Сформировать прогноз", key="predict_button"):

        if not os.path.exists("models/ag_model_v1"):
            st.error("Сначала обучите модель 👈")
            st.stop()

    try:
        with st.spinner("Расчет прогноза..."):
            model.load()
            predictions = model.predict(ts_data)

            fig = plot_forecast(ts_data, predictions, selected_country)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"Данные прогноза для {selected_country}")
            st.write(predictions.loc[selected_country])

    except Exception as e:
        st.error(f"Ошибка: {e}")

    except Exception as e:
        st.error(f"Сначала обучите модель или проверьте путь к ней. Ошибка: {e}")

except FileNotFoundError:
    st.error(
        f"Файл не найден по пути {DATA_PATH}. Пожалуйста, скачайте датасет с Kaggle и положите его в папку data/raw/"
    )

# Футер с метриками (опционально)
if st.checkbox("Показать Leaderboard моделей"):
    model.load()
    st.dataframe(model.get_leaderboard())
