# Позволяет обращаться к модулям напрямую через пакет src
from .data_loader import prepare_timeseries_data
from .model import ForecastModel
from .visualize import plot_forecast

__all__ = ["prepare_timeseries_data", "ForecastModel", "plot_forecast"]
