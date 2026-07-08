# Sustainable Energy Consumption Prediction (LSTM)

Predicting residential energy usage from smart meter time-series data using an LSTM model built in PyTorch.

## Problem Statement

Predict residential energy usage based on smart meter time-series data.
**Dataset:** UCI Household Power Consumption

## Dataset

- Source: [UCI Machine Learning Repository — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- File: `household_power_consumption.txt`
- ~2,075,259 rows of minute-level readings collected between December 2006 and November 2010 from a single household in Sceaux, France
- Semicolon-separated, missing values marked as `?`

### Columns used
| Column | Description |
|---|---|
| Date, Time | Combined into a single `Datetime` index |
| Global_active_power | Household global minute-averaged active power (kW) — this is the target variable |

## Approach

1. **Load & clean** — read the raw text file, parse `Date` + `Time` into a proper datetime index, forward-fill missing readings.
2. **Resample** — average the minute-level data into hourly readings, since raw per-minute data is too large and noisy to model directly.
3. **Scale** — normalize the `Global_active_power` values to a 0–1 range with `MinMaxScaler`, since LSTMs train more reliably on scaled inputs.
4. **Windowing** — build sliding windows of the past 24 hours to predict the next hour's usage.
5. **Train/test split** — chronological 80/20 split (no shuffling, since order matters for time series).
6. **Model** — a single-layer LSTM (`hidden_size=64`) followed by a fully connected layer that outputs one predicted value.
7. **Training** — MSE loss, Adam optimizer, trained for 15 epochs.
8. **Evaluation** — predictions are inverse-transformed back to real kW values and compared against actuals using RMSE and MAE.
9. **Forecasting** — a helper function takes the most recent 24 hours of readings and predicts the next hour's power consumption.

## Model Architecture

```
Input (24 timesteps, 1 feature)
        ↓
    LSTM (hidden_size=64)
        ↓
  Fully Connected (64 → 1)
        ↓
   Predicted next-hour power (kW)
```

## Results

| Metric | Value |
|---|---|
| RMSE | *fill in from your run* |
| MAE | *fill in from your run* |

*(See the training loss plot and the actual-vs-predicted plot in the notebook for details.)*

## Tech Stack

- Python
- pandas / NumPy — data loading and preprocessing
- scikit-learn — MinMax scaling
- PyTorch — LSTM model and training loop
- Matplotlib — visualization

## How to Run

1. Download the UCI Household Power Consumption dataset and place `household_power_consumption.txt` in the project folder.
2. Update the file path in the data-loading cell if needed.
3. Run the notebook cells in order — from data loading through training, evaluation, and next-hour prediction.

## Possible Improvements

- Use a larger window size or additional features (e.g. sub-metering columns, voltage) as extra input channels
- Try a stacked or bidirectional LSTM, or compare against GRU
- Add early stopping / a validation split during training
- Forecast multiple hours ahead instead of just the next hour
