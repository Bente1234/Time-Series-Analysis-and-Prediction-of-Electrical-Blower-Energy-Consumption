import os
import random
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ForecastTrainer:
    """Handles train/test split, model fitting, and rolling forecasts."""

    def __init__(self, config: Dict):
        self.config = config
        self.forecasting_config = config["forecasting"]
        self.models_config = config["models"]
        self.lstm_config = config["lstm"]

        seed = config["project"]["random_seed"]

        np.random.seed(seed)
        random.seed(seed)

        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    def split_data(self, df_15min: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Split time series into training and testing sets while preserving order."""
        df = df_15min.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        split_ratio = self.forecasting_config["train_ratio"]
        split_idx = int(len(df) * split_ratio)

        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()

        train_ts = train.set_index("timestamp")["Consumption_kWh"].asfreq("15min")
        test_ts = test.set_index("timestamp")["Consumption_kWh"].asfreq("15min")

        return train_ts, test_ts

    def fit_models(self, train_ts: pd.Series) -> Dict:
        """Fit MA, Exponential Smoothing, AR, ARIMA, and LSTM models."""
        results = {}

        print("\n--- Exercise 4(ii): Build Forecasting Models ---")
        print("Training series frequency:", train_ts.index.freq)
        print("Missing values in training series:", train_ts.isna().sum())

        if train_ts.isna().sum() > 0:
            raise ValueError("Training series contains missing values after setting frequency.")

        best_ma_aic = np.inf
        best_ma_order = None
        best_ma_model = None

        for q in range(1, self.models_config["ma_max_q"] + 1):
            try:
                model = ARIMA(train_ts, order=(0, 0, q))
                fitted = model.fit()
                if fitted.aic < best_ma_aic:
                    best_ma_aic = fitted.aic
                    best_ma_order = (0, 0, q)
                    best_ma_model = fitted
            except Exception:
                continue

        print(f"Best MA model: ARIMA{best_ma_order} with AIC = {best_ma_aic:.2f}")

        results["MA"] = {
            "model": best_ma_model,
            "order": best_ma_order,
            "aic": best_ma_aic,
        }

        es_model = ExponentialSmoothing(
            train_ts,
            trend=self.models_config["exponential_smoothing"]["trend"],
            seasonal=self.models_config["exponential_smoothing"]["seasonal"],
        )
        es_fitted = es_model.fit()

        print("Exponential Smoothing model fitted")

        results["ExpSmoothing"] = {
            "model": es_fitted,
            "type": "Simple Exponential Smoothing",
        }

        best_ar_aic = np.inf
        best_ar_lag = None
        best_ar_model = None

        for p in range(1, self.models_config["ar_max_p"] + 1):
            try:
                model = AutoReg(train_ts, lags=p, old_names=False)
                fitted = model.fit()
                if fitted.aic < best_ar_aic:
                    best_ar_aic = fitted.aic
                    best_ar_lag = p
                    best_ar_model = fitted
            except Exception:
                continue

        print(f"Best AR model: AR({best_ar_lag}) with AIC = {best_ar_aic:.2f}")

        results["AR"] = {
            "model": best_ar_model,
            "lag": best_ar_lag,
            "aic": best_ar_aic,
        }

        best_arima_aic = np.inf
        best_arima_order = None
        best_arima_model = None

        for p in range(self.models_config["arima_max_p"] + 1):
            for q in range(self.models_config["arima_max_q"] + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(train_ts, order=(p, 0, q))
                    fitted = model.fit()
                    if fitted.aic < best_arima_aic:
                        best_arima_aic = fitted.aic
                        best_arima_order = (p, 0, q)
                        best_arima_model = fitted
                except Exception:
                    continue

        print(f"Best ARIMA model: ARIMA{best_arima_order} with AIC = {best_arima_aic:.2f}")

        results["ARIMA"] = {
            "model": best_arima_model,
            "order": best_arima_order,
            "aic": best_arima_aic,
        }

        if self.lstm_config["enabled"]:
            from sklearn.preprocessing import MinMaxScaler
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping
            from tensorflow.keras.layers import LSTM, Dense, Input
            from tensorflow.keras.models import Sequential

            tf.random.set_seed(self.config["project"]["random_seed"])

            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_ts.values.reshape(-1, 1))

            def create_sequences(data, window_size):
                X, y = [], []
                for i in range(window_size, len(data)):
                    X.append(data[i - window_size:i, 0])
                    y.append(data[i, 0])
                return np.array(X), np.array(y)

            window_size = self.lstm_config["window_size"]
            X_train, y_train = create_sequences(train_scaled, window_size)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

            print("LSTM training input shape:", X_train.shape)
            print("LSTM target shape:", y_train.shape)

            lstm_model = Sequential([
                Input(shape=(window_size, 1)),
                LSTM(self.lstm_config["lstm_units"]),
                Dense(1)
            ])

            lstm_model.compile(optimizer="adam", loss="mse")

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=self.lstm_config["patience"],
                restore_best_weights=True
            )

            lstm_history = lstm_model.fit(
                X_train,
                y_train,
                epochs=self.lstm_config["epochs"],
                batch_size=self.lstm_config["batch_size"],
                validation_split=self.lstm_config["validation_split"],
                callbacks=[early_stop],
                verbose=1
            )

            print("LSTM model trained")

            results["LSTM"] = {
                "model": lstm_model,
                "scaler": scaler,
                "window_size": window_size,
                "history": lstm_history,
                "input_shape": X_train.shape,
                "target_shape": y_train.shape,
            }

        print("\n--- Selected Models ---")
        print(f"MA model: ARIMA{results['MA']['order']}")
        print("Exponential Smoothing: Simple Exponential Smoothing")
        print(f"AR model: AR({results['AR']['lag']})")
        print(f"ARIMA model: ARIMA{results['ARIMA']['order']}")
        if "LSTM" in results:
            print(f"LSTM window size: {results['LSTM']['window_size']}")

        return results

    def generate_rolling_forecasts(
        self,
        train_ts: pd.Series,
        test_ts: pd.Series,
        models: Dict,
    ) -> pd.DataFrame:
        """Generate rolling multi-step forecasts and keep the final horizon step."""
        print("\n--- Exercise 4(iii): Rolling 6-step-ahead Forecast ---")
        print("Testing series frequency:", test_ts.index.freq)
        print("Missing values in testing series:", test_ts.isna().sum())

        if test_ts.isna().sum() > 0:
            raise ValueError("Testing series contains missing values after setting frequency.")

        horizon = self.forecasting_config["horizon"]
        step_size = self.forecasting_config["step_size"]
        refit_every = self.forecasting_config["refit_every"]

        print("Rolling forecast horizon:", horizon)
        print("Step size:", step_size)
        print("Refit every:", refit_every, "rolling origins")

        forecast_index = []
        actual_values = []

        ma_preds = []
        es_preds = []
        ar_preds = []
        arima_preds = []
        lstm_preds = []

        full_series = pd.concat([train_ts, test_ts])

        ma_order = models["MA"]["order"]
        ar_lag = models["AR"]["lag"]
        arima_order = models["ARIMA"]["order"]

        lstm_model = models["LSTM"]["model"] if "LSTM" in models else None
        lstm_scaler = models["LSTM"]["scaler"] if "LSTM" in models else None
        lstm_window = models["LSTM"]["window_size"] if "LSTM" in models else None

        ma_fitted_roll = None
        es_fitted_roll = None
        ar_fitted_roll = None
        arima_fitted_roll = None

        rolling_positions = list(range(0, len(test_ts) - horizon + 1, step_size))

        for j, i in enumerate(rolling_positions):
            history = full_series.iloc[: len(train_ts) + i]

            if (j % refit_every == 0) or (ma_fitted_roll is None):
                ma_fitted_roll = ARIMA(history, order=ma_order).fit()
                es_fitted_roll = ExponentialSmoothing(history, trend=None, seasonal=None).fit()
                ar_fitted_roll = AutoReg(history, lags=ar_lag, old_names=False).fit()
                arima_fitted_roll = ARIMA(history, order=arima_order).fit()

            target_time = test_ts.index[i + horizon - 1]
            actual_value = test_ts.iloc[i + horizon - 1]

            forecast_index.append(target_time)
            actual_values.append(actual_value)

            ma_preds.append(ma_fitted_roll.forecast(steps=horizon).iloc[-1])
            es_preds.append(es_fitted_roll.forecast(steps=horizon).iloc[-1])
            ar_preds.append(ar_fitted_roll.forecast(steps=horizon).iloc[-1])
            arima_preds.append(arima_fitted_roll.forecast(steps=horizon).iloc[-1])

            if lstm_model is not None:
                history_scaled = lstm_scaler.transform(history.values.reshape(-1, 1))
                current_window = history_scaled[-lstm_window:].copy()

                next_pred_scaled = None
                for _ in range(horizon):
                    x_input = current_window.reshape((1, lstm_window, 1))
                    next_pred_scaled = lstm_model.predict(x_input, verbose=0)[0, 0]
                    current_window = np.vstack([current_window[1:], [[next_pred_scaled]]])

                lstm_pred = lstm_scaler.inverse_transform([[next_pred_scaled]])[0, 0]
                lstm_preds.append(lstm_pred)

            if (j + 1) % 10 == 0:
                print(f"Processed {j + 1}/{len(rolling_positions)} rolling forecasts")

        forecast_df = pd.DataFrame(
            {
                "actual": actual_values,
                "MA": ma_preds,
                "ExpSmoothing": es_preds,
                "AR": ar_preds,
                "ARIMA": arima_preds,
                "LSTM": lstm_preds,
            },
            index=forecast_index,
        )

        print("\nForecast dataframe preview:")
        print(forecast_df.head())

        print("\nForecast dataframe shape:", forecast_df.shape)
        print("Forecast period:")
        print("Start:", forecast_df.index.min())
        print("End:  ", forecast_df.index.max())

        return forecast_df