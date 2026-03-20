import plotly.graph_objects as go
import pandas as pd

def plot_forecast(train_data, predictions, item_id, title="Прогноз выбросов CO2"):
    """
    Рисует интерактивный график: история + прогноз с доверительными интервалами.
    
    :param train_data: Исходный TimeSeriesDataFrame (история)
    :param predictions: Результат работы predictor.predict() (прогноз)
    :param item_id: Название страны (item_id), которую хотим отобразить
    """
    # 1. Извлекаем историю для конкретной страны
    history = train_data.loc[item_id]
    
    # 2. Извлекаем прогноз для этой же страны
    # AutoGluon выдает среднее значение (mean) и границы (0.1, 0.9 квантили)
    forecast = predictions.loc[item_id]
    
    fig = go.Figure()

    # Линия исторических данных
    fig.add_trace(go.Scatter(
        x=history.index, 
        y=history['target'],
        name='История',
        line=dict(color='royalblue', width=3)
    ))

    # Линия прогноза (среднее значение)
    fig.add_trace(go.Scatter(
        x=forecast.index, 
        y=forecast['mean'],
        name='Прогноз (Mean)',
        line=dict(color='firebrick', width=3, dash='dash')
    ))

    # Доверительный интервал (заливка между 10% и 90% вероятности)
    fig.add_trace(go.Scatter(
        x=forecast.index.tolist() + forecast.index.tolist()[::-1],
        y=forecast['0.9'].tolist() + forecast['0.1'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 0, 0, 0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Доверительный интервал (80%)'
    ))

    fig.update_layout(
        title=f"{title}: {item_id}",
        xaxis_title="Год",
        yaxis_title="Выбросы CO2 (тонн)",
        hovermode="x unified",
        template="plotly_white"
    )

    return fig
