#About Page

import dash
from dash import html

dash.register_page(__name__, path='/about', title='Anomaly Detection')

layout = html.Div([
    html.H1('About:')
])