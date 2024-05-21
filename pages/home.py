#Home Page

import dash
from dash import html

dash.register_page(__name__, path='/', title='Anomaly Detection Dashboard')

layout = html.Div([
    html.H1('Home Page for Time Series Anomaly Detection'),
    html.P("Select one of the above options to either:"),
    html.Ul([
        html.Li('Generate plots and do anomaly detection on time series data.'),
        html.Li('Learn more about the anomaly detection technique.')
    ])
])