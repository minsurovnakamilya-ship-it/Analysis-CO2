import plotly.graph_objects as go
import pandas as pd


def plot_forecast(train_data, predictions, item_id, title="Прогноз выбросов CO2"):
    """
    Рисует интерактивный график: история + прогноз с доверительными интервалами.
    """
    # 1. Извлекаем историю и прогноз для конкретной страны
    history = train_data.loc[item_id]
    forecast = predictions.loc[item_id]

    # Определяем колонки квантилей (они могут быть '0.1' или 0.1)
    q_low = "0.1" if "0.1" in forecast.columns else 0.1
    q_high = "0.9" if "0.9" in forecast.columns else 0.9

    fig = go.Figure()

    # Линия исторических данных
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["target"],
            name="История",
            line=dict(color="#1f77b4", width=3),  # Красивый синий
        )
    )

    # Доверительный интервал (Заливка) - ДОЛЖНА БЫТЬ ПЕРЕД ЛИНИЕЙ ПРОГНОЗА
    # Чтобы линия прогноза была сверху заливки
    fig.add_trace(
        go.Scatter(
            x=list(forecast.index) + list(forecast.index)[::-1],
            y=list(forecast[q_high]) + list(forecast[q_low])[::-1],
            fill="toself",
            fillcolor="rgba(214, 39, 40, 0.2)",  # Полупрозрачный красный
            line=dict(color="rgba(255, 255, 255, 0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Интервал неопределенности (80%)",
        )
    )

    # Линия прогноза (среднее значение)
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["mean"],
            name="Прогноз",
            line=dict(color="#d62728", width=3, dash="dash"),  # Красный пунктир
        )
    )

    # Настройка внешнего вида
    fig.update_layout(
        title=dict(
            text=f"<b>{title}: {item_id}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        xaxis_title="Год",
        yaxis_title="Выбросы CO2",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=80, b=20),
        height=600,
    )

    # Включаем сетку для удобства
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")

    return fig
