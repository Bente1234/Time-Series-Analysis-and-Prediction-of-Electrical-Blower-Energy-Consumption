from typing import Dict, Tuple

import pandas as pd


class TimeSeriesPreprocessor:
    """Handles timestamp creation, cleaning, feature engineering, and 15-minute aggregation."""

    def __init__(self, config: Dict):
        self.config = config
        self.preprocessing_config = config["preprocessing"]

        self.date_column = self.preprocessing_config["timestamp_date_column"]
        self.time_column = self.preprocessing_config["timestamp_time_column"]
        self.consumption_column = self.preprocessing_config["consumption_column"]
        self.original_index_column = self.preprocessing_config["original_index_column"]
        self.renamed_index_column = self.preprocessing_config["renamed_index_column"]
        self.timestamp_format = self.preprocessing_config["timestamp_format"]
        self.off_threshold = self.preprocessing_config["off_threshold"]
        self.resample_frequency = self.preprocessing_config["resample_frequency"]

    def prepare_raw_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data by renaming the original index column, creating a timestamp,
        sorting chronologically, handling missing consumption values, and creating
        a binary machine_on feature.
        """
        df = raw_df.copy()

        if self.original_index_column in df.columns:
            df = df.rename(columns={self.original_index_column: self.renamed_index_column})

        datetime_string = (
            df[self.date_column].astype(str).str.strip()
            + " "
            + df[self.time_column].astype(str).str.strip()
        )

        df["timestamp"] = pd.to_datetime(
            datetime_string,
            format=self.timestamp_format,
            errors="coerce",
        )

        df = df.sort_values("timestamp").reset_index(drop=True)

        df[self.consumption_column] = df[self.consumption_column].fillna(0.0)
        df["machine_on"] = (df[self.consumption_column] >= self.off_threshold).astype(int)

        return df

    def compute_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute descriptive statistics required for early assignment tasks."""
        total_observations = len(df)
        invalid_timestamps = df["timestamp"].isna().sum()
        duplicate_timestamps = df["timestamp"].duplicated().sum()

        time_diffs = df["timestamp"].diff().dropna()
        average_interval = time_diffs.mean()
        average_interval_minutes = average_interval.total_seconds() / 60 if not time_diffs.empty else None

        off_mask = (df[self.consumption_column] < self.off_threshold) | (df[self.consumption_column].isna())
        num_off = off_mask.sum()
        percent_off = (num_off / total_observations) * 100 if total_observations > 0 else 0

        return {
            "total_observations": total_observations,
            "invalid_timestamps": invalid_timestamps,
            "duplicate_timestamps": duplicate_timestamps,
            "average_interval": average_interval,
            "average_interval_minutes": average_interval_minutes,
            "num_off": num_off,
            "percent_off": percent_off,
        }

    def build_15min_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert irregular interval energy observations into a 15-minute energy series
        by proportionally allocating consumption across overlapping 15-minute bins.
        """
        working_df = df.copy()
        working_df = working_df.sort_values("timestamp").reset_index(drop=True)
        working_df["timestamp"] = pd.to_datetime(working_df["timestamp"])

        working_df["prev_timestamp"] = working_df["timestamp"].shift(1)

        if not working_df.empty:
            working_df.loc[working_df.index[0], "prev_timestamp"] = working_df.loc[
                working_df.index[0], "timestamp"
            ].floor("D")

        working_df = working_df.dropna(subset=["prev_timestamp"]).copy()

        working_df["dt_seconds"] = (
            working_df["timestamp"] - working_df["prev_timestamp"]
        ).dt.total_seconds()

        working_df["power"] = working_df[self.consumption_column] / working_df["dt_seconds"]

        records = []

        for _, row in working_df.iterrows():
            start = row["prev_timestamp"]
            end = row["timestamp"]
            power = row["power"]

            bins = pd.date_range(
                start.floor(self.resample_frequency),
                end.ceil(self.resample_frequency),
                freq=self.resample_frequency,
            )

            for bin_start, bin_end in zip(bins[:-1], bins[1:]):
                overlap_start = max(start, bin_start)
                overlap_end = min(end, bin_end)
                overlap_seconds = (overlap_end - overlap_start).total_seconds()

                if overlap_seconds > 0:
                    energy = power * overlap_seconds
                    records.append((bin_start, energy))

        df_15min = (
            pd.DataFrame(records, columns=["timestamp", "Consumption_kWh"])
            .groupby("timestamp", as_index=False)["Consumption_kWh"]
            .sum()
        )

        df_15min = df_15min.sort_values("timestamp").reset_index(drop=True)
        return df_15min