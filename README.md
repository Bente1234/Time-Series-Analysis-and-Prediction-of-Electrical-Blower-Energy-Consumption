# 1BM170 Assignment 2 - Group 11

## Project overview
This project analyzes and forecasts industrial machine energy consumption using time series methods. The workflow covers:
- data understanding and time index construction,
- preprocessing and stationarity testing,
- pattern analysis and feature extraction,
- forecasting and model evaluation.

The implementation is written in Python and organized in a modular, object-oriented structure.

## Project structure

```text
1BM170 Assignment 2 - Group 11/
├── config/
│   └── config.yaml
├── data/
│   ├── KwhConsumptionBlower78_1.csv
│   ├── KwhConsumptionBlower78_2.csv
│   └── KwhConsumptionBlower78_3.csv
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── forecaster.py
│   └── preprocessor.py
├── main.py
├── requirements.txt
└── README.md
```


## Installation Instructions

### Install dependancies
```bash
pip install -r requirements.txt
pip install tensorflow
```

### Run the project
```bash
python main.py
```