# Revenue Forecasting for Digital Health HMO

This project forecasts future revenue for a Digital Health HMO company using time series analysis.

## Features
- 5-year revenue projection with 95% confidence intervals
- Model evaluation using Mean Absolute Percentage Error (MAPE)
- Interactive visualization of historical and forecasted revenue
- Business insights generation including growth rates

## Usage
1. Install requirements: `pip install -r ../requirements.txt`
2. Place your data in `data/digital_health_hmo.csv` with columns: `year,revenue`
3. Run the script: `python revenue_forecasting.py`

## Outputs
- `output/forecast_results.csv`: Forecast data in CSV format
- `output/forecast_visualization.html`: Interactive Plotly visualization

## Example Insights
"Based on historical trends, revenue is projected to grow at an average rate of 12% annually over the next 5 years, with a range of 8-16% at 95% confidence level."
