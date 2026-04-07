import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from src.data_loader import prepare_timeseries_data
from src.model import ForecastModel
from src.visualize import plot_forecast
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


st.set_page_config(page_title="CO2 Forecast System", layout="wide")
st.title("Система прогнозирования выбросов CO2")

DATA_PATH = "data/processed/owid_co2_data.csv"


@st.cache_data
def load_data():
    return prepare_timeseries_data(DATA_PATH)


try:
    ts_data = load_data()
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Базовый прогноз",
            "Прогноз с признаками",
            "Аналитика",
            "Карта CO2",
        ]
    )

    countries = ts_data.index.get_level_values(0).unique().tolist()

    st.sidebar.header("Настройки прогноза")
    selected_country = st.sidebar.selectbox(
        "Выберите страну/регион",
        countries,
        index=countries.index("World") if "World" in countries else 0,
    )
    prediction_years = st.sidebar.slider("Горизонт прогноза (лет)", 5, 30, 10)

   
    with tab1:
        model_base = ForecastModel(model_path="models/ag_model_v1")

        if st.sidebar.button("Обучить базовую модель", key="train_button"):
            with st.spinner("Обучение базовой модели..."):
                model_base.train(ts_data, prediction_length=prediction_years)
            st.sidebar.success("Базовая модель обучена!")

        if st.button("Сформировать базовый прогноз", key="predict_button"):
            if not os.path.exists("models/ag_model_v1"):
                st.error("Сначала обучите модель 👈")
            else:
                model_base.load()
                predictions = model_base.predict(ts_data)
                fig = plot_forecast(ts_data, predictions, selected_country)
                st.plotly_chart(fig, use_container_width=True)

    
    with tab2:
        st.header("Прогноз с дополнительными признаками")
        st.info("Модель учитывает GDP, Population и Energy для поиска зависимостей.")

        cov_model_path = "models/ag_model_cov"

        if st.button("Обучить модель с признаками", key="train_cov"):
            with st.spinner("Обучение модели с признаками..."):
                predictor_cov = TimeSeriesPredictor(
                    prediction_length=prediction_years,
                    path=cov_model_path,
                    target="target",
                    freq="Y",
                )
                predictor_cov.fit(ts_data, presets="medium_quality", time_limit=300)
            st.success("Модель с признаками обучена!")

        if st.button("Сформировать прогноз (с признаками)", key="predict_cov"):
            if not os.path.exists(cov_model_path):
                st.error("Сначала обучите модель с признаками!")
            else:
                try:
                    predictor_cov = TimeSeriesPredictor.load(cov_model_path)
                    predictions = predictor_cov.predict(ts_data)
                    fig = plot_forecast(ts_data, predictions, selected_country)
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(predictions.loc[selected_country])
                except Exception as e:
                    st.error(f"Ошибка при прогнозировании: {e}")

  
    with tab3:
        st.header("Аналитика данных")

        df = pd.DataFrame(ts_data).reset_index()

        countries = df["item_id"].unique().tolist()
        selected_country_analytics = st.selectbox(
            "Выберите страну для анализа",
            countries,
            index=countries.index("World") if "World" in countries else 0,
            key="analytics_country",
        )

        
        st.subheader("Динамика выбросов CO2")
        country_df = df[df["item_id"] == selected_country_analytics]
        fig1, ax1 = plt.subplots()
        ax1.plot(country_df["timestamp"], country_df["target"])
        ax1.set_title(f"CO2 выбросы: {selected_country_analytics}")
        ax1.set_xlabel("Год")
        ax1.set_ylabel("CO2")
        st.pyplot(fig1)

        st.subheader("Корреляция признаков")
        cols = ["target", "gdp", "population", "primary_energy_consumption"]
        cols = [c for c in cols if c in df.columns]

        if len(cols) > 2:
            corr = df[cols].corr()
            fig2, ax2 = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)
        else:
            st.warning("Недостаточно признаков для корреляции")

        st.subheader("Влияние факторов на CO2")
        if "target" in df.columns:
            corr_target = df[cols].corr()["target"].sort_values(ascending=False)
            st.dataframe(corr_target)
            st.bar_chart(corr_target)

        
        st.subheader("Топ-10 стран по выбросам CO2")
        top_countries = df.groupby("item_id")["target"].max().nlargest(10).reset_index()
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x="target", y="item_id", data=top_countries, palette="magma", ax=ax3
        )
        ax3.set_xlabel("Максимальный CO2")
        ax3.set_ylabel("Страна / Регион")
        st.pyplot(fig3)


    with tab4:
        st.header("Глобальная карта выбросов CO2")
        df = ts_data.reset_index()
        latest_df = df.sort_values("timestamp").groupby("item_id").last().reset_index()
        latest_df = latest_df[latest_df["item_id"] != "World"]  # исключаем глобум

        fig_map = go.Figure(
            data=go.Choropleth(
                locations=latest_df["item_id"],
                locationmode="country names",
                z=latest_df["target"],
                colorscale="Reds",
                colorbar_title="CO2",
            )
        )

        fig_map.update_layout(
            geo=dict(showframe=False, showcoastlines=True),
            margin=dict(l=0, r=0, t=0, b=0),
        )

        st.plotly_chart(fig_map, use_container_width=True)

   
    if st.checkbox("Показать сравнение моделей"):
        try:
            current_model = ForecastModel(model_path="models/ag_model_v1")

            if not os.path.exists("models/ag_model_v1"):
                st.warning("Сначала обучите модель")
            else:
                current_model.load()
                st.subheader("Сравнение моделей (AutoML)")
                leaderboard = current_model.get_detailed_leaderboard()

                if leaderboard is not None:
                    st.dataframe(leaderboard)
                    rmse = current_model.evaluate_rmse(ts_data)
                    st.write(f"Средний RMSE: {rmse:.3f}")
                    st.bar_chart(leaderboard.set_index("Model")["MASE"])
                else:
                    st.warning("Нет данных о моделях")
        except Exception as e:
            st.error(f"Ошибка при загрузке лидерборда: {e}")

except FileNotFoundError:
    st.error(f"Файл не найден по пути {DATA_PATH}. Проверьте папку data/processed/")
except Exception as e:
    st.error(f"Произошла ошибка при загрузке данных: {e}")
