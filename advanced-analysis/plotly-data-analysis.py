import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc


years = list(range(2017, 2040))
categories = ['Digital Treatment & Care', 'Online Doctor Consultations', 
              'Digital Fitness & Well-Being', 'Water Resources Institute']

revenue_data = {
    'Digital Treatment & Care': [30.00, 50.31, 64.97, 91.95, 114.51, 132.93, 147.97, 171.86, 191.88, 
                                218.81, 226.31, 248.98, 254.25],
    'Online Doctor Consultations': [26.73, 33.29, 41.32, 60.31, 58.53, 34.02, 36.40, 100.15, 122.03, 
                                   136.61, 143.93, 156.78, 158.31],
    'Digital Fitness & Well-Being': [0.79, 0.87, 1.36, 3.10, 8.87, 7.19, 8.56, 8.46, 10.14, 
                                    10.64, 11.20, 11.31, 11.55],
    'Water Resources Institute': [13.57, 16.80, 22.31, 28.41, 36.40, 41.71, 50.01, 58.64, 66.73, 
                                 71.68, 76.37, 80.28, 83.43]
}

revenue_df = pd.DataFrame(revenue_data, index=years[:len(revenue_data['Digital Treatment & Care'])])
revenue_df.index.name = 'Year'

users_data = {
    'Total': [364.54, 463.18, 589.11, 780.94, 916.98, 1036.34, 1186.61, 1329.87, 1462.70, 
              1570.45, 1652.74, 1718.99, 1774.11],
    'Online Doctor Consultations': [27.70, 42.16, 59.80, 84.27, 101.86, 106.31, 116.10, 122.39, 
                                   127.17, 131.69, 135.48, 138.36, 140.88],
    'Digital Treatment & Care': [366.90, 427.11, 494.60, 584.78, 647.49, 706.83, 790.76, 880.86, 
                                979.78, 1062.90, 1129.54, 1185.52, 1237.76],
    'Digital Fitness & Well-Being': [199.01, 283.96, 402.28, 598.04, 738.76, 869.25, 1019.31, 
                                    1155.35, 1267.38, 1354.83, 1418.35, 1467.45, 1502.59]
}

users_df = pd.DataFrame(users_data, index=years[:len(users_data['Total'])])
users_df.index.name = 'Year'


penetration_data = {
    'Total': [5.00, 6.29, 7.92, 10.41, 12.12, 13.59, 15.44, 17.16, 18.72, 
              19.94, 20.82, 21.49, 22.01],
    'Digital Treatment & Care': [5.04, 5.80, 6.65, 7.79, 8.56, 9.27, 10.29, 11.37, 12.54, 
                                13.50, 14.23, 14.82, 15.36],
    'Online Doctor Consultations': [0.38, 0.57, 0.80, 1.12, 1.35, 1.39, 1.51, 1.58, 1.63, 
                                   1.67, 1.71, 1.73, 1.75],
    'Digital Fitness & Well-Being': [2.73, 3.86, 5.41, 7.97, 9.76, 11.40, 13.26, 14.91, 16.22, 
                                    17.20, 17.87, 18.35, 18.64]
}

penetration_df = pd.DataFrame(penetration_data, index=years[:len(penetration_data['Total'])])
penetration_df.index.name = 'Year'

arpu_df = revenue_df.copy()
for col in arpu_df.columns:
    if col in users_df.columns:
        arpu_df[col] = revenue_df[col] * 1e9 / (users_df[col] * 1e6)  # Convert to per-user USD

correlation_data = []
for category in ['Digital Treatment & Care', 'Online Doctor Consultations', 'Digital Fitness & Well-Being']:
    corr = np.corrcoef(revenue_df[category], users_df[category])[0,1]
    correlation_data.append({'Category': category, 'Correlation': corr})
correlation_df = pd.DataFrame(correlation_data)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Digital Health Market Analysis"

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Digital Health Market Analysis Dashboard"), className="mb-4")),
    
    dbc.Row([
        dbc.Col([
            html.H3("Market Segment Selection"),
            dcc.Dropdown(
                id='category-selector',
                options=[{'label': cat, 'value': cat} for cat in 
                         ['Digital Treatment & Care', 'Online Doctor Consultations', 'Digital Fitness & Well-Being']],
                value='Digital Treatment & Care',
                clearable=False
            ),
            html.Br(),
            html.H3("Analysis Metric"),
            dcc.RadioItems(
                id='metric-selector',
                options=[
                    {'label': 'Revenue vs Users', 'value': 'rev_users'},
                    {'label': 'ARPU Trends', 'value': 'arpu'},
                    {'label': 'Penetration Impact', 'value': 'penetration'},
                    {'label': 'Growth Rates', 'value': 'growth'}
                ],
                value='rev_users',
                labelStyle={'display': 'block'}
            )
        ], md=3),
        
        dbc.Col(dcc.Graph(id='main-graph'), md=9)
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='correlation-heatmap'), md=6),
        dbc.Col(dcc.Graph(id='market-drivers'), md=6)
    ]),
    
    dbc.Row([
        dbc.Col(html.Div(id='insights-container', className="mt-4 p-3 bg-light"))
    ])
], fluid=True)

@app.callback(
    [Output('main-graph', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('market-drivers', 'figure'),
     Output('insights-container', 'children')],
    [Input('category-selector', 'value'),
     Input('metric-selector', 'value')]
)
def update_graphs(selected_category, selected_metric):
    # Main graph based on selected metric
    if selected_metric == 'rev_users':
        fig_main = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add revenue trace
        fig_main.add_trace(
            go.Scatter(
                x=revenue_df.index,
                y=revenue_df[selected_category],
                name=f"{selected_category} Revenue",
                line=dict(color='royalblue', width=3)
            ),
            secondary_y=False
        )
        
        # Add users trace
        fig_main.add_trace(
            go.Scatter(
                x=users_df.index,
                y=users_df[selected_category],
                name=f"{selected_category} Users",
                line=dict(color='firebrick', width=3, dash='dot')
            ),
            secondary_y=True
        )
        
        fig_main.update_layout(
            title=f"{selected_category}: Revenue vs User Growth",
            xaxis_title="Year",
            yaxis_title="Revenue (Billion USD)",
            yaxis2_title="Users (Million)",
            hovermode="x unified"
        )
        
    elif selected_metric == 'arpu':
        fig_main = go.Figure()
        fig_main.add_trace(
            go.Scatter(
                x=arpu_df.index,
                y=arpu_df[selected_category],
                name=f"{selected_category} ARPU",
                line=dict(color='green', width=3)
            )
        )
        fig_main.update_layout(
            title=f"{selected_category}: Average Revenue Per User (ARPU)",
            xaxis_title="Year",
            yaxis_title="ARPU (USD)",
            hovermode="x"
        )
        
    elif selected_metric == 'penetration':
        fig_main = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add revenue trace
        fig_main.add_trace(
            go.Scatter(
                x=revenue_df.index,
                y=revenue_df[selected_category],
                name=f"{selected_category} Revenue",
                line=dict(color='royalblue', width=3)
            ),
            secondary_y=False
        )
        
        # Add penetration trace
        fig_main.add_trace(
            go.Scatter(
                x=penetration_df.index,
                y=penetration_df[selected_category],
                name=f"{selected_category} Penetration",
                line=dict(color='purple', width=3, dash='dot')
            ),
            secondary_y=True
        )
        
        fig_main.update_layout(
            title=f"{selected_category}: Revenue vs Market Penetration",
            xaxis_title="Year",
            yaxis_title="Revenue (Billion USD)",
            yaxis2_title="Penetration Rate (%)",
            hovermode="x unified"
        )
        
    elif selected_metric == 'growth':
        # Calculate growth rates
        revenue_growth = revenue_df[selected_category].pct_change() * 100
        users_growth = users_df[selected_category].pct_change() * 100
        
        fig_main = go.Figure()
        fig_main.add_trace(
            go.Scatter(
                x=revenue_growth.index[1:],
                y=revenue_growth[1:],
                name="Revenue Growth",
                line=dict(color='royalblue', width=3)
            )
        )
        fig_main.add_trace(
            go.Scatter(
                x=users_growth.index[1:],
                y=users_growth[1:],
                name="User Growth",
                line=dict(color='firebrick', width=3, dash='dot')
            )
        )
        fig_main.update_layout(
            title=f"{selected_category}: Revenue vs User Growth Rates",
            xaxis_title="Year",
            yaxis_title="Growth Rate (%)",
            hovermode="x unified"
        )
    
    # Correlation heatmap
    corr_matrix = pd.concat([
        revenue_df[['Digital Treatment & Care', 'Online Doctor Consultations', 'Digital Fitness & Well-Being']],
        users_df[['Digital Treatment & Care', 'Online Doctor Consultations', 'Digital Fitness & Well-Being']]
    ], axis=1).corr()
    
    fig_heatmap = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Viridis',
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        hoverinfo="text"
    ))
    fig_heatmap.update_layout(
        title="Revenue-User Correlation Matrix",
        xaxis_title="Metrics",
        yaxis_title="Metrics"
    )
    
    # Market drivers chart
    drivers = {
        'Driver': ['Capitalized uncertainty', 'GEO Growth', 'Liberality', 
                   'Telemedicine trend', 'CEPT population', 
                   'Technology advances', 'Health spending', 'COVID-19 trends'],
        'Impact 2023': [11.3, -1.3, 1.0, 1.3, 1.5, 1.2, 0.9, 6.7]
    }
    drivers_df = pd.DataFrame(drivers)
    
    fig_drivers = go.Figure(go.Bar(
        x=drivers_df['Impact 2023'],
        y=drivers_df['Driver'],
        orientation='h',
        marker_color=['red' if x < 0 else 'green' for x in drivers_df['Impact 2023']]
    ))
    fig_drivers.update_layout(
        title="Market Value Drivers (2023 Impact in %)",
        xaxis_title="Impact on Market Value Change (%)",
        yaxis_title="Driver"
    )
    
    # Generate insights
    corr_value = correlation_df[correlation_df['Category'] == selected_category]['Correlation'].values[0]
    
    if corr_value > 0.8:
        corr_strength = "very strong positive"
    elif corr_value > 0.6:
        corr_strength = "strong positive"
    elif corr_value > 0.4:
        corr_strength = "moderate positive"
    else:
        corr_strength = "weak"
    
    if selected_category == 'Digital Treatment & Care':
        insights = [
            html.H4("Key Insights: Digital Treatment & Care"),
            html.P(f"Revenue and users show {corr_strength} correlation (r={corr_value:.2f})."),
            html.P("This segment demonstrates consistent growth in both users and revenue, indicating a healthy market expansion."),
            html.P("The ARPU has been increasing, suggesting higher monetization per user over time."),
            html.P("Market penetration is steadily increasing, reaching 15.36% by 2039."),
            html.P("The strong correlation suggests user acquisition directly drives revenue in this segment.")
        ]
    elif selected_category == 'Online Doctor Consultations':
        insights = [
            html.H4("Key Insights: Online Doctor Consultations"),
            html.P(f"Revenue and users show {corr_strength} correlation (r={corr_value:.2f})."),
            html.P("This segment shows interesting dynamics with revenue growth outpacing user growth in recent years."),
            html.P("The significant revenue jump in 2024 suggests a business model change or market consolidation."),
            html.P("ARPU has been volatile, indicating potential pricing strategy changes."),
            html.P("Penetration remains relatively low (1.75% by 2039), suggesting significant growth potential.")
        ]
    elif selected_category == 'Digital Fitness & Well-Being':
        insights = [
            html.H4("Key Insights: Digital Fitness & Well-Being"),
            html.P(f"Revenue and users show {corr_strength} correlation (r={corr_value:.2f})."),
            html.P("This segment shows signs of market maturation with slowing growth rates in both users and revenue."),
            html.P("ARPU has remained relatively stable, suggesting a mature pricing model."),
            html.P("Penetration reaches 18.64% by 2039, indicating widespread adoption."),
            html.P("The market may be approaching saturation, with growth coming from existing user monetization.")
        ]
    
    return fig_main, fig_heatmap, fig_drivers, insights

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)