import os
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor
from sklearn.metrics import mean_squared_error
import shutil


class ForecastModel:
    def __init__(self, model_path="models/ag_model_v1"):
        self.model_path = model_path
        self.predictor = None

    def train(self, train_data, prediction_length=10, presets="medium_quality"):
        """Обучение моделей."""
        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path)

        self.predictor = TimeSeriesPredictor(
            path=self.model_path,
            prediction_length=prediction_length,
            target="target",
            eval_metric="MASE",
            freq="Y",
        )

        self.predictor.fit(train_data, presets=presets, time_limit=300)  # 5 мин
        print(f"Модель обучена и сохранена в {self.model_path}")

    def load(self):
        """Загрузка обученной модели."""
        if os.path.exists(self.model_path):
            self.predictor = TimeSeriesPredictor.load(self.model_path)
        else:
            raise FileNotFoundError(f"Модель не найдена по пути {self.model_path}")

    def predict(self, train_data):
        """Прогноз с автоматической обработкой признаков."""
        if self.predictor is None:
            self.load()

        try:
            return self.predictor.predict(train_data)
        except Exception as e:
            if "known_covariates" in str(e):
                future_cov = self.predictor.construct_empty_future_covariates(
                    train_data
                )

                for item_id in train_data.item_ids:
                    last_known_row = train_data.loc[item_id].iloc[-1:]
                    for col in future_cov.columns:
                        if col in last_known_row.columns:
                            val = last_known_row[col].values[0]
                            future_cov.loc[(item_id, slice(None)), col] = val

                return self.predictor.predict(train_data, known_covariates=future_cov)
            else:
                raise e

    def get_detailed_leaderboard(self):
        """Красивая таблица результатов AutoML."""
        if self.predictor:
            lb = self.predictor.leaderboard()
            cols = ["model", "score_val", "fit_time_marginal", "pred_time_val"]
            available = [c for c in cols if c in lb.columns]

            lb = lb[available].rename(
                columns={
                    "model": "Model",
                    "score_val": "MASE",
                    "fit_time_marginal": "Train Time (s)",
                    "pred_time_val": "Predict Time (s)",
                }
            )
            return lb
        return None

    def evaluate_rmse(self, train_data):
        """Оценка точности на исторических данных."""
        if self.predictor is None:
            self.load()

        predictions = self.predict(train_data)
        rmses = []

        items = train_data.item_ids[:20]
        for item_id in items:
            try:
                true = train_data.loc[item_id]["target"].values[
                    -len(predictions.loc[item_id]) :
                ]
                pred = predictions.loc[item_id]["mean"].values
                if len(true) == len(pred):
                    rmse = np.sqrt(mean_squared_error(true, pred))
                    rmses.append(rmse)
            except:
                continue
        return np.mean(rmses) if rmses else 0

    def get_detailed_leaderboard(self):
        if self.predictor:
            lb = self.predictor.leaderboard()

            columns_to_keep = [
                col
                for col in ["model", "score_val", "fit_time_marginal", "pred_time_val"]
                if col in lb.columns
            ]

            lb = lb[columns_to_keep]

            lb = lb.rename(
                columns={
                    "model": "Model",
                    "score_val": "MASE",
                    "fit_time_marginal": "Train Time (s)",
                    "pred_time_val": "Predict Time (s)",
                }
            )

            return lb

        return None
