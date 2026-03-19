from pathlib import Path

from src.analyzer import TimeSeriesAnalyzer
from src.config import ConfigLoader
from src.data_loader import DataLoader
from src.evaluator import ForecastEvaluator
from src.forecaster import ForecastTrainer
from src.preprocessor import TimeSeriesPreprocessor


def main():
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config" / "config.yaml"

    config = ConfigLoader(config_path).load()

    data_loader = DataLoader(config)
    preprocessor = TimeSeriesPreprocessor(config)
    analyzer = TimeSeriesAnalyzer(config)
    forecaster = ForecastTrainer(config)
    evaluator = ForecastEvaluator(config)

    raw_df = data_loader.load_raw_data()
    clean_df = preprocessor.prepare_raw_data(raw_df)
    stats = preprocessor.compute_basic_statistics(clean_df)
    df_15min = preprocessor.build_15min_series(clean_df)
    analysis_results = analyzer.run_full_analysis(df_15min)

    adf_results = analysis_results["adf_results"]
    peak_low_results = analysis_results["peak_low_results"]
    mean_variance_results = analysis_results["mean_variance_results"]
    hourly_pattern = analysis_results["hourly_pattern"]

    print("---------")
    print("Exercise 1: Data Understanding and Time Index Construction")
    print("---------\n")
    print(f"Total observations: {stats['total_observations']}")
    print(f"Average time interval: {stats['average_interval']}")
    print(f"Average time interval (minutes): {stats['average_interval_minutes']:.2f}")
    print(f"OFF observations: {stats['num_off']}")
    print(f"OFF percentage: {stats['percent_off']:.2f}%")
    print(f"Invalid timestamps: {stats['invalid_timestamps']}")
    print(f"Duplicate timestamps: {stats['duplicate_timestamps']}")

    print("\nPreview of cleaned raw data:")
    print(clean_df.head(5).to_string(index=False))

    print("\n---------")
    print("Exercise 2: Time Series Preprocessing and Statistical Analysis")
    print("---------\n")
    print("15-minute dataframe shape:", df_15min.shape)

    print("\nADF test results:")
    print(f"ADF Statistic: {adf_results['adf_statistic']:.6f}")
    print(f"p-value: {adf_results['p_value']:.12f}")
    print(f"Lags used: {adf_results['lags_used']}")
    print(f"Number of observations: {adf_results['n_observations']}")
    print("Critical values:")
    for key, value in adf_results["critical_values"].items():
        print(f"  {key}: {value:.6f}")

    print("\n---------")
    print("Exercise 3: Pattern Analysis and Feature Extraction")
    print("---------\n")
    print("Peak and low consumption thresholds:")
    print(f"Peak threshold (90th percentile): {peak_low_results['peak_threshold']:.4f}")
    print(f"Low threshold (10th percentile): {peak_low_results['low_threshold']:.4f}")
    print(f"Number of peak periods: {peak_low_results['num_peak_periods']}")
    print(f"Number of low periods: {peak_low_results['num_low_periods']}")

    print("\nMean and variance statistics:")
    print(f"Mean consumption when machine is ON: {mean_variance_results['mean_consumption_on']:.4f}")
    print(f"Variance of consumption (all periods): {mean_variance_results['variance_all']:.4f}")
    print(f"Variance of consumption (ON periods): {mean_variance_results['variance_on']:.4f}")

    print("\nHourly pattern preview:")
    print(hourly_pattern.head(10).round(4).to_string())

    train_ts, test_ts = forecaster.split_data(df_15min)

    print("\n---------")
    print("Exercise 4: Time Series Forecasting and Evaluation")
    print("---------\n")
    print("--- Exercise 4(i): Train-Test Split ---")
    print(f"Total observations: {len(df_15min)}")
    print(f"Training observations: {len(train_ts)} ({len(train_ts) / len(df_15min) * 100:.1f}%)")
    print(f"Testing observations: {len(test_ts)} ({len(test_ts) / len(df_15min) * 100:.1f}%)")

    print("\nTraining period:")
    print(f"Start: {train_ts.index.min()}")
    print(f"End:   {train_ts.index.max()}")

    print("\nTesting period:")
    print(f"Start: {test_ts.index.min()}")
    print(f"End:   {test_ts.index.max()}")

    print("\nTemporal order preserved:", train_ts.index.max() < test_ts.index.min())

    models = forecaster.fit_models(train_ts)
    forecast_df = forecaster.generate_rolling_forecasts(train_ts, test_ts, models)
    results_df = evaluator.evaluate(forecast_df)
    evaluator.plot_forecasts(forecast_df)

    print("\n--- MODEL TRAINING COMPLETE ---")
    for name in models.keys():
        print(f"{name} model trained")

    print("\nSaved evaluation table:")
    print(results_df.round(4).to_string(index=False))

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()