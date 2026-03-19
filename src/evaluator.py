from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ForecastEvaluator:
    """Evaluates forecasting performance and saves outputs."""

    def __init__(self, config: Dict):
        self.config = config
        self.output_config = config["output"]

        self.project_root = Path(__file__).resolve().parent.parent
        self.figures_dir = self.project_root / config["paths"]["figures_dir"]
        self.tables_dir = self.project_root / config["paths"]["tables_dir"]

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def evaluate(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        print("\n--- Exercise 4(iv): Forecast Evaluation ---")
        print("Missing values per column in forecast_df:")
        print(forecast_df.isna().sum())

        if forecast_df.isna().any().any():
            raise ValueError("forecast_df contains missing values. Evaluation cannot proceed.")

        models = ["MA", "ExpSmoothing", "AR", "ARIMA", "LSTM"]
        results = []

        for model in models:
            y_true = forecast_df["actual"]
            y_pred = forecast_df[model]

            results.append(
                {
                    "Model": model,
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "RMSE": self.rmse(y_true, y_pred),
                    "MAPE (%)": self.mape(y_true, y_pred),
                }
            )

        results_df = pd.DataFrame(results).sort_values(by="MAE").reset_index(drop=True)

        print("\nForecast accuracy results:")
        print(results_df.round(4))

        best_mae_model = results_df.loc[results_df["MAE"].idxmin(), "Model"]
        best_rmse_model = results_df.loc[results_df["RMSE"].idxmin(), "Model"]
        best_mape_model = results_df.loc[results_df["MAPE (%)"].idxmin(), "Model"]

        print("\nBest model by MAE:", best_mae_model)
        print("Best model by RMSE:", best_rmse_model)
        print("Best model by MAPE:", best_mape_model)

        if self.output_config["save_tables"]:
            results_df.to_csv(self.tables_dir / "forecast_results.csv", index=False)

        return results_df

    def plot_forecasts(self, forecast_df: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df.index, forecast_df["actual"], label="Actual", linewidth=2)
        plt.plot(forecast_df.index, forecast_df["ExpSmoothing"], label="Exp. Smoothing")
        plt.plot(forecast_df.index, forecast_df["MA"], label="MA")
        plt.plot(forecast_df.index, forecast_df["AR"], label="AR")
        plt.plot(forecast_df.index, forecast_df["ARIMA"], label="ARIMA")
        plt.plot(forecast_df.index, forecast_df["LSTM"], label="LSTM")

        plt.title("Energy Consumption Forecasts vs Actual Values (Test Period)")
        plt.xlabel("Time")
        plt.ylabel("Energy Consumption (kWh)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if self.output_config["save_figures"]:
            plt.savefig(self.figures_dir / "forecast_comparison.png", dpi=300, bbox_inches="tight")

        if self.output_config["show_figures"]:
            plt.show()
        else:
            plt.close()