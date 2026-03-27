import os
import streamlit as st
import pandas as pd
from src.data_loader import prepare_timeseries_data
from src.model import ForecastModel
from src.visualize import plot_forecast
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# Настройка страницы
st.set_page_config(page_title="CO2 Forecast System", layout="wide")
st.title("🌍 Система прогнозирования выбросов CO2")

DATA_PATH = "data/processed/owid_co2_data.csv"


@st.cache_data
def load_data():
    return prepare_timeseries_data(DATA_PATH)


try:
    ts_data = load_data()
    tab1, tab2 = st.tabs(["📈 Базовый прогноз", "🧠 Прогноз с признаками"])
    countries = ts_data.index.get_level_values(0).unique().tolist()

    st.sidebar.header("Настройки прогноза")
    selected_country = st.sidebar.selectbox(
        "Выберите страну/регион",
        countries,
        index=countries.index("World") if "World" in countries else 0,
    )
    prediction_years = st.sidebar.slider("Горизонт прогноза (лет)", 5, 30, 10)
    # --- ТАБ 1: БАЗОВЫЙ ПРОГНОЗ ---
    with tab1:
        model_base = ForecastModel(model_path="models/ag_model_v1")

        if st.sidebar.button("Обучить базовую модель", key="train_button"):
            with st.spinner("Обучение базовой модели..."):
                # Важно: передаем весь ts_data, но модель сама возьмет только 'target'
                # так как мы не указывали ковариаты при инициализации
                model_base.train(ts_data, prediction_length=prediction_years)
            st.sidebar.success("Базовая модель обучена!")

        if st.button("Сформировать базовый прогноз", key="predict_button"):
            if not os.path.exists("models/ag_model_v1"):
                st.error("Сначала обучите модель 👈")
            else:
                model_base.load()
                # Предсказываем по тем же данным
                predictions = model_base.predict(ts_data)
                fig = plot_forecast(ts_data, predictions, selected_country)
                st.plotly_chart(fig, use_container_width=True)

    # --- ТАБ 2: ПРОГНОЗ С ПРИЗНАКАМИ ---
    with tab2:
        st.header("🧠 Прогноз с дополнительными признаками")
        st.info("Модель учитывает GDP, Population и Energy для поиска зависимостей.")

        cov_model_path = "models/ag_model_cov"

        if st.button("Обучить модель с признаками", key="train_cov"):
            with st.spinner("Обучение модели с признаками..."):
                # Убираем known_covariates_names, чтобы модель воспринимала их как Past Covariates
                # Это позволит делать прогноз, не зная будущего ВВП
                predictor_cov = TimeSeriesPredictor(
                    prediction_length=prediction_years,
                    path=cov_model_path,
                    target="target",
                    freq="Y",
                )
                # Обучаем на полных данных (с ВВП и прочим)
                predictor_cov.fit(ts_data, presets="medium_quality", time_limit=300)
            st.success("Модель с признаками обучена!")

        if st.button("Сформировать прогноз (с признаками)", key="predict_cov"):
            if not os.path.exists(cov_model_path):
                st.error("Сначала обучите модель с признаками!")
            else:
                try:
                    predictor_cov = TimeSeriesPredictor.load(cov_model_path)

                    # Если модель всё же просит ковариаты, создаем "заглушку" на будущее
                    try:
                        predictions = predictor_cov.predict(ts_data)
                    except Exception:
                        # Создаем пустую таблицу для будущего и заполняем её последними значениями
                        future_cov = predictor_cov.construct_empty_future_covariates(
                            ts_data
                        )
                        # Заполняем пропуски в будущем последними известными значениями (ffill)
                        for item in ts_data.item_ids:
                            last_vals = ts_data.loc[item].iloc[-1]
                            for col in future_cov.columns:
                                future_cov.loc[(item, slice(None)), col] = last_vals[
                                    col
                                ]

                        predictions = predictor_cov.predict(
                            ts_data, known_covariates=future_cov
                        )

                    fig = plot_forecast(ts_data, predictions, selected_country)
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(predictions.loc[selected_country])

                except Exception as e:
                    st.error(f"Ошибка: {e}")

except Exception as e:
    st.error(f"Критическая ошибка: {e}")

except FileNotFoundError:
    st.error(f"Файл не найден по пути {DATA_PATH}. Проверьте папку data/processed/")
except Exception as e:
    st.error(f"Произошла ошибка при загрузке данных: {e}")

if st.checkbox("📊 Показать сравнение моделей"):
    try:
        # Здесь мы заново создаем объект для этого блока
        current_model = ForecastModel(model_path="models/ag_model_v1")

        if not os.path.exists("models/ag_model_v1"):
            st.warning("Сначала обучите модель")
        else:
            current_model.load()
            st.subheader("Сравнение моделей (AutoML)")
            leaderboard = current_model.get_detailed_leaderboard()

            if leaderboard is not None:
                st.dataframe(leaderboard)
                # Передаем ts_data целиком
                rmse = current_model.evaluate_rmse(ts_data)
                st.write(f"📉 Средний RMSE: {rmse:.3f}")
                st.bar_chart(leaderboard.set_index("Model")["MASE"])
            else:
                st.warning("Нет данных о моделях")
    except Exception as e:
        st.error(f"Ошибка при загрузке лидерборда: {e}")
