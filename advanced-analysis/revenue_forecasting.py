import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

load_dotenv()
plt.style.use('seaborn')
pd.set_option('display.float_format', '{:,.2f}'.format)

class RevenueForecaster:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.forecast_periods = 5  # Forecast 5 years ahead
        self.confidence_level = 0.95  # 95% confidence interval
        
    def preprocess_data(self):
        """Prepare the revenue time series data"""
        df = self.data.copy()
        df['year'] = pd.to_datetime(df['year'], format='%Y')
        df.set_index('year', inplace=True)
        return df['revenue']
    
    def train_model(self, series):
        """Train Holt-Winters forecasting model"""
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal='add',
            seasonal_periods=4,
            damped_trend=True
        )
        self.model_fit = model.fit()
        return self.model_fit
    
    def generate_forecast(self, series):
        """Generate forecast with confidence intervals"""
        forecast = self.model_fit.get_forecast(
            steps=self.forecast_periods,
            alpha=1-self.confidence_level
        )
        
        future_dates = pd.date_range(
            start=series.index[-1] + pd.DateOffset(years=1),
            periods=self.forecast_periods,
            freq='Y'
        )
        
        forecast_df = pd.DataFrame({
            'year': future_dates.year,
            'forecast': forecast.predicted_mean,
            'lower': forecast.conf_int()['lower revenue'],
            'upper': forecast.conf_int()['upper revenue']
        })
        
        return forecast_df
    
    def evaluate_model(self, series):
        """Evaluate model performance with MAPE"""
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        
        model = ExponentialSmoothing(
            train,
            trend='add',
            seasonal='add',
            seasonal_periods=4
        ).fit()
        
        predictions = model.forecast(len(test))
        mape = mean_absolute_percentage_error(test, predictions) * 100
        return mape
    
    def visualize_forecast(self, historical, forecast):
        """Create interactive forecast visualization"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical,
            name='Historical Revenue',
            line=dict(color='royalblue', width=3)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast['year'],
            y=forecast['forecast'],
            name='Forecast',
            line=dict(color='green', width=3, dash='dot')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['year'].tolist() + forecast['year'].tolist()[::-1],
            y=forecast['upper'].tolist() + forecast['lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(
            title='Digital Health HMO Revenue Forecast (Next 5 Years)',
            xaxis_title='Year',
            yaxis_title='Revenue (Billion USD)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def generate_insights(self, forecast, mape):
        """Generate business insights from forecast"""
        avg_growth = ((forecast['forecast'].iloc[-1] / forecast['forecast'].iloc[0]) ** (1/5) - 1) * 100
        min_growth = ((forecast['lower'].iloc[-1] / forecast['lower'].iloc[0]) ** (1/5) - 1) * 100
        max_growth = ((forecast['upper'].iloc[-1] / forecast['upper'].iloc[0]) ** (1/5) - 1) * 100
        
        insights = {
            'average_growth': avg_growth,
            'min_growth': min_growth,
            'max_growth': max_growth,
            'mape': mape,
            'next_year_forecast': forecast['forecast'].iloc[0],
            'next_year_range': (
                forecast['lower'].iloc[0],
                forecast['upper'].iloc[0]
            )
        }
        
        return insights
    
    def run_analysis(self):
        """Run complete forecasting pipeline"""
        series = self.preprocess_data()
        self.train_model(series)
        forecast = self.generate_forecast(series)
        mape = self.evaluate_model(series)
        fig = self.visualize_forecast(series, forecast)
        insights = self.generate_insights(forecast, mape)
        
        # Save outputs
        forecast.to_csv('output/forecast_results.csv', index=False)
        fig.write_html('output/forecast_visualization.html')
        
        print(f"Forecast completed with MAPE: {mape:.2f}%")
        print(f"Average annual growth projection: {insights['average_growth']:.2f}%")
        
        return insights

if __name__ == "__main__":
    forecaster = RevenueForecaster('data/digital_health_hmo.csv')
    results = forecaster.run_analysis()
