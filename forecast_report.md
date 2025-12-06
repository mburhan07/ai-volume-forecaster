# 📊 Quantitative AI Forecasting Report

## 1. Overview
For this project I generated historical volume data and used statistical analysis to identify patterns in the time series. I then implemented two forecasting methods: a moving average and exponential smoothing. The analysis also included decomposition of the time series and visualizations for weekly and monthly seasonality.

---

## 2. Historical Data
The dataset contained **over 1000 daily observations** from **2021–2023**.  
The `data_generator.py` script created the CSV file:

- Path: `data/historical_volumes.csv`
- Fields: `date`, `volume`

---

## 3. Statistical Pattern Analysis

Key metrics from the dataset:

- Mean (μ): **~1390**
- Standard Deviation (σ): **~192**
- Coefficient of Variation: **~0.138**
- Linear Trend: slightly positive (upward)

These values show that the volume fluctuates daily but remains consistent over time.

Weekly pattern results:

- Certain days (example: **Tue, Wed, Thu**) had higher average volume than others.
- Visualization highlighted recurring behavior through the week.

---

## 4. Forecasting Results

### Moving Average (Window = 7)
A 7-day moving average was calculated and used as a forecast.

- **Next period forecast:** approximately **the most recent MA value**
- **Historical error (MAE):** low, meaning this method captured the patterns well

---

### Exponential Smoothing (α = 0.3)
This method weights recent observations more heavily.

- **Next period forecast:** similar to moving average
- **Historical error (MAE):** also low and close to the moving average error

Both methods produced consistent results.

---

## 5. Time Series Decomposition

The series was decomposed into:

- **Trend:** gradual upward direction
- **Seasonal:** repeating weekly and monthly patterns
- **Residual:** random noise

This confirmed the presence of structure in the data and regular recurring patterns.

---

## 6. Visualizations

The script generated **volume_analysis.png** automatically.

It includes:

- Actual volume vs 7-day moving average
- Weekly average bar chart
- Monthly average trend line

All plots were saved to the project root.

---

## 7. Conclusion

The system successfully:

✔ Generated historical data  
✔ Performed statistical pattern analysis  
✔ Produced forecasts using moving average and exponential smoothing  
✔ Decomposed the time series  
✔ Created visualizations and saved them  
✔ Printed a complete analysis report  

This forms the **foundation of a Quantitative AI Forecasting System**.  
In Week 8, machine learning models will be added to improve accuracy and prediction performance.

---

## 8. Repository

All code, data, and results are committed to the public repository.
Project completed on: <today's date>)
