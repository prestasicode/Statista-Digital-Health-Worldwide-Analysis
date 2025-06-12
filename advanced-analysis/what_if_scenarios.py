import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from dotenv import load_dotenv
import statsmodels.api as sm

load_dotenv()
plt.style.use('seaborn')
pd.set_option('display.float_format', '{:,.2f}'.format)

class ScenarioAnalyzer:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.scenarios = {
            'optimistic': 1.10,  # 10% better than baseline
            'baseline': 1.00,    # No change
            'pessimistic': 0.90  # 10% worse than baseline
        }
    
    def preprocess_data(self):
        """Prepare data for scenario analysis"""
        df = self.data.copy()
        
        df['gdp_growth'] = df['gdp'].pct_change() * 100
        df['internet_penetration'] = df['internet_users'] / df['population'] * 100
        df['revenue_growth'] = df['revenue'].pct_change() * 100
        
        df['gdp_growth_lag1'] = df['gdp_growth'].shift(1)
        df['internet_penetration_lag1'] = df['internet_penetration'].shift(1)
        
        return df.dropna()
    
    def train_model(self, data):
        """Train predictive model for revenue growth"""
        X = data[['gdp_growth_lag1', 'internet_penetration_lag1']]
        y = data['revenue_growth']
        
        # OLS model for interpretability
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        
        return model
    
    def generate_scenarios(self, data, model):
        """Generate revenue projections under different scenarios"""
        last_values = data.iloc[-1][['gdp_growth', 'internet_penetration']]
        
        scenarios_df = pd.DataFrame()
        
        for name, multiplier in self.scenarios.items():
            scenario_data = {
                'gdp_growth': last_values['gdp_growth'] * multiplier,
                'internet_penetration': last_values['internet_penetration'] * multiplier
            }
            
            projection = []
            current_revenue = data['revenue'].iloc[-1]
            
            for year in range(1, 6):
                # Predict growth rate
                X_pred = pd.DataFrame({
                    'const': [1],
                    'gdp_growth_lag1': [scenario_data['gdp_growth']],
                    'internet_penetration_lag1': [scenario_data['internet_penetration']]
                })
                
                growth_rate = model.predict(X_pred)[0]
                
                # Apply growth to revenue
                current_revenue *= (1 + growth_rate / 100)
                projection.append(current_revenue)
                
                # Update scenario factors with some mean reversion
                scenario_data['gdp_growth'] *= 0.95 if multiplier > 1 else 1.05
                scenario_data['internet_penetration'] *= 1.02  # Slight continual increase
            
            scenarios_df[name] = projection
        
        scenarios_df['year'] = range(data['year'].max() + 1, data['year'].max() + 6)
        return scenarios_df
    
    def visualize_scenarios(self, historical, scenarios):
        """Create interactive scenario comparison"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical['year'],
            y=historical['revenue'],
            name='Historical Revenue',
            line=dict(color='grey', width=2, dash='dot')
        ))
        
        colors = {'optimistic': 'green', 'baseline': 'blue', 'pessimistic': 'red'}
        
        for scenario in scenarios.columns[:-1]:  # Exclude year column
            fig.add_trace(go.Scatter(
                x=scenarios['year'],
                y=scenarios[scenario],
                name=scenario.capitalize(),
                line=dict(color=colors[scenario], width=3)
            ))
        
        fig.update_layout(
            title='Digital Health HMO Revenue Under Different Scenarios',
            xaxis_title='Year',
            yaxis_title='Revenue (Billion USD)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def generate_insights(self, scenarios, model):
        """Generate business insights from scenarios"""
        # Calculate differences
        optimistic = scenarios['optimistic'].iloc[-1]
        pessimistic = scenarios['pessimistic'].iloc[-1]
        difference = optimistic - pessimistic
        percent_diff = (difference / pessimistic) * 100
        
        # Get factor impacts from model
        gdp_impact = model.params['gdp_growth_lag1']
        internet_impact = model.params['internet_penetration_lag1']
        
        insights = {
            'revenue_range_2028': (pessimistic, optimistic),
            'potential_difference': difference,
            'percent_difference': percent_diff,
            'gdp_sensitivity': gdp_impact,
            'internet_sensitivity': internet_impact,
            'recommendations': self._generate_recommendations(model)
        }
        
        return insights
    
    def _generate_recommendations(self, model):
        """Generate strategic recommendations based on model"""
        gdp_impact = model.params['gdp_growth_lag1']
        internet_impact = model.params['internet_penetration_lag1']
        
        recs = []
        
        if gdp_impact > 1.0:
            recs.append("Diversify across economic cycles - revenue is highly sensitive to GDP growth")
        if internet_impact > 0.5:
            recs.append("Invest in digital infrastructure partnerships - internet penetration drives growth")
        if abs(gdp_impact) < 0.5 and abs(internet_impact) < 0.5:
            recs.append("Focus on product differentiation - external factors have limited impact")
        
        return recs
    
    def run_analysis(self):
        """Run complete scenario analysis pipeline"""
        data = self.preprocess_data()
        model = self.train_model(data)
        scenarios = self.generate_scenarios(data, model)
        fig = self.visualize_scenarios(data, scenarios)
        insights = self.generate_insights(scenarios, model)
        
        scenarios.to_csv('output/scenario_results.csv', index=False)
        fig.write_html('output/scenario_visualization.html')
        with open('output/scenario_insights.txt', 'w') as f:
            for key, value in insights.items():
                if key == 'recommendations':
                    f.write("\nRecommendations:\n")
                    for rec in value:
                        f.write(f"- {rec}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        print("Scenario analysis completed")
        print(f"2028 Revenue Range: ${insights['revenue_range_2028'][0]:.2f}B - ${insights['revenue_range_2028'][1]:.2f}B")
        
        return insights

if __name__ == "__main__":
    analyzer = ScenarioAnalyzer('data/digital_health_hmo_full.csv')
    insights = analyzer.run_analysis()
