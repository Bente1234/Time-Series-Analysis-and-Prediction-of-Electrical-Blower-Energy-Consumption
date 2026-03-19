from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller


class TimeSeriesAnalyzer:
    """Runs statistical analysis and creates visualizations for the 15-minute energy series."""

    def __init__(self, config: Dict):
        self.config = config
        self.analysis_config = config["analysis"]
        self.output_config = config["output"]

        self.project_root = Path(__file__).resolve().parent.parent
        self.figures_dir = self.project_root / config["paths"]["figures_dir"]
        self.tables_dir = self.project_root / config["paths"]["tables_dir"]

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def run_full_analysis(self, df_15min: pd.DataFrame) -> Dict:
        """Run all descriptive analyses and return summary results."""
        analysis_df = df_15min.copy()
        analysis_df["timestamp"] = pd.to_datetime(analysis_df["timestamp"])
        analysis_df = analysis_df.sort_values("timestamp").reset_index(drop=True)

        daily_consumption = self.compute_daily_consumption(analysis_df)
        hourly_pattern = self.compute_hourly_pattern(analysis_df)
        moving_average = self.compute_daily_moving_average(daily_consumption)

        adf_results = self.run_adf_test(analysis_df["Consumption_kWh"])
        peak_low_results = self.compute_peak_low_periods(analysis_df)
        mean_variance_results = self.compute_mean_and_variance(analysis_df)

        self.plot_energy_consumption_over_time(analysis_df)
        self.plot_daily_aggregated_consumption(daily_consumption)
        self.plot_hourly_average_consumption(hourly_pattern)
        self.plot_daily_moving_average(daily_consumption, moving_average)
        self.plot_daily_acf(daily_consumption)

        return {
            "daily_consumption": daily_consumption,
            "hourly_pattern": hourly_pattern,
            "moving_average": moving_average,
            "adf_results": adf_results,
            "peak_low_results": peak_low_results,
            "mean_variance_results": mean_variance_results,
        }

    def compute_daily_consumption(self, df_15min: pd.DataFrame) -> pd.Series:
        working_df = df_15min.copy()
        working_df["date"] = working_df["timestamp"].dt.date
        return working_df.groupby("date")["Consumption_kWh"].sum()

    def compute_hourly_pattern(self, df_15min: pd.DataFrame) -> pd.Series:
        working_df = df_15min.copy()
        working_df["hour"] = working_df["timestamp"].dt.hour
        return working_df.groupby("hour")["Consumption_kWh"].mean()

    def compute_daily_moving_average(self, daily_consumption: pd.Series) -> pd.Series:
        window = self.analysis_config["moving_average_window_days"]
        center = self.analysis_config["moving_average_center"]
        return daily_consumption.rolling(window=window, center=center).mean()

    def run_adf_test(self, series: pd.Series) -> Dict:
        adf_result = adfuller(series)
        return {
            "adf_statistic": adf_result[0],
            "p_value": adf_result[1],
            "lags_used": adf_result[2],
            "n_observations": adf_result[3],
            "critical_values": adf_result[4],
        }

    def compute_peak_low_periods(self, df_15min: pd.DataFrame) -> Dict:
        peak_quantile = self.analysis_config["peak_quantile"]
        low_quantile = self.analysis_config["low_quantile"]

        peak_threshold = df_15min["Consumption_kWh"].quantile(peak_quantile)
        low_threshold = df_15min["Consumption_kWh"].quantile(low_quantile)

        peak_periods = df_15min[df_15min["Consumption_kWh"] >= peak_threshold]
        low_periods = df_15min[df_15min["Consumption_kWh"] <= low_threshold]

        return {
            "peak_threshold": peak_threshold,
            "low_threshold": low_threshold,
            "num_peak_periods": len(peak_periods),
            "num_low_periods": len(low_periods),
        }

    def compute_mean_and_variance(self, df_15min: pd.DataFrame) -> Dict:
        working_df = df_15min.copy()
        working_df["machine_on"] = (working_df["Consumption_kWh"] >= 1).astype(int)

        mean_consumption_on = working_df.loc[
            working_df["machine_on"] == 1, "Consumption_kWh"
        ].mean()

        variance_all = working_df["Consumption_kWh"].var()
        variance_on = working_df.loc[
            working_df["machine_on"] == 1, "Consumption_kWh"
        ].var()

        return {
            "mean_consumption_on": mean_consumption_on,
            "variance_all": variance_all,
            "variance_on": variance_on,
        }

    def plot_energy_consumption_over_time(self, df_15min: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 5))
        plt.plot(df_15min["timestamp"], df_15min["Consumption_kWh"])
        plt.title("Energy Consumption Over Time")
        plt.xlabel("Time")
        plt.ylabel("Energy Consumption (kWh)")
        plt.tight_layout()
        self._finalize_figure("energy_consumption_over_time.png")

    def plot_daily_aggregated_consumption(self, daily_consumption: pd.Series) -> None:
        plt.figure(figsize=(12, 5))
        plt.plot(daily_consumption.index, daily_consumption.values)
        plt.title("Daily Aggregated Energy Consumption")
        plt.xlabel("Date")
        plt.ylabel("Total Daily Consumption (kWh)")
        plt.tight_layout()
        self._finalize_figure("daily_aggregated_consumption.png")

    def plot_hourly_average_consumption(self, hourly_pattern: pd.Series) -> None:
        plt.figure(figsize=(12, 5))
        plt.plot(hourly_pattern.index, hourly_pattern.values)
        plt.title("Average Consumption by Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Average Consumption (kWh)")
        plt.tight_layout()
        self._finalize_figure("hourly_average_consumption.png")

    def plot_daily_moving_average(
        self,
        daily_consumption: pd.Series,
        moving_average: pd.Series,
    ) -> None:
        plt.figure(figsize=(12, 5))
        plt.plot(daily_consumption.index, daily_consumption.values, label="Daily Consumption")
        plt.plot(moving_average.index, moving_average.values, label="7-Day Moving Average")
        plt.title("Daily Energy Consumption with Moving Average")
        plt.xlabel("Date")
        plt.ylabel("Energy Consumption (kWh)")
        plt.legend()
        plt.tight_layout()
        self._finalize_figure("daily_moving_average.png")

    def plot_daily_acf(self, daily_consumption: pd.Series) -> None:
        days_to_show = self.analysis_config["acf_days_to_show"]
        daily_series = daily_consumption[-days_to_show:]

        fig, ax = plt.subplots(figsize=(12, 5))
        plot_acf(daily_series.values, lags=days_to_show - 1, ax=ax, alpha=0.05)

        ax.set_xticks(np.arange(0, days_to_show + 1, 7))
        ax.set_xticklabels([f"Day {i}" for i in range(0, days_to_show + 1, 7)])
        ax.xaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_title(
            "Autocorrelation Function (ACF) — Daily Aggregated Series, Confidence Level 95%"
        )
        plt.tight_layout()
        self._finalize_figure("acf_daily_consumption.png")

    def _finalize_figure(self, filename: str) -> None:
        if self.output_config["save_figures"]:
            output_path = self.figures_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        if self.output_config["show_figures"]:
            plt.show()
        else:
            plt.close()
            