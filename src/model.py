import os
from autogluon.timeseries import TimeSeriesPredictor

class ForecastModel:
    def __init__(self, model_path="models/ag_model_v1"):
        """
        Инициализация модели.
        :param model_path: Путь, куда AutoGluon сохранит обученные артефакты.
        """
        self.model_path = model_path
        self.predictor = None

    def train(self, train_data, prediction_length=10, presets="medium_quality"):
        """
        Обучение моделей на данных о выбросах.
        :param prediction_length: На сколько лет вперед делать прогноз (горизонт).
        :param presets: Качество обучения ('fast_training', 'medium_quality', 'high_quality').
        """
        self.predictor = TimeSeriesPredictor(
            prediction_length=prediction_length,
            path=self.model_path,
            target="target",
            eval_metric="MASE", 
             freq="Y"  # Хорошая метрика для временных рядов
        )
        
        # Обучаем ансамбль моделей (AutoGluon сам выберет лучшие)
        self.predictor.fit(
            train_data,
            presets=presets,
            time_limit=600  # Ограничение по времени в секундах (например, 10 мин)
        )
        print(f"Модель успешно обучена и сохранена в {self.model_path}")

    def load(self):
        """Загрузка уже обученной модели с диска."""
        if os.path.exists(self.model_path):
            self.predictor = TimeSeriesPredictor.load(self.model_path)
            print("Модель загружена.")
        else:
            raise FileNotFoundError("Папка с моделью не найдена. Сначала вызовите train().")

    def predict(self, train_data):
        """Получение прогноза."""
        if self.predictor is None:
            self.load()
        
        # Делаем предсказание на заданный при обучении горизонт
        predictions = self.predictor.predict(train_data)
        return predictions

    def get_leaderboard(self):
        """Показать таблицу эффективности разных моделей внутри AutoGluon."""
        if self.predictor:
            return self.predictor.leaderboard()
        return None
