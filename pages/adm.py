#ADM (Anomaly Detection Menu)

import dash
from dash import html
from dash import dcc
import pandas as pd
from DashDataAPI import apimain
import plotly.express as px
import plotly.graph_objects as go

dash.register_page(__name__, path='/adm', title='Anomaly Detection Dashboard')

layout = html.Div([
    html.H1('Anomaly Detection Menu'),
    html.Div([
        html.Label('Model:'),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'Majority Vote', 'value': 'multi'},
                {'label': 'Generative Labeling', 'value': 'gen'},
                {'label': 'DBSCAN Clustering', 'value': 'dbscan'},
                {'label': 'KNN', 'value': 'knn'},
                {'label': 'One Class SVM', 'value': 'svm'},
                {'label': 'Local Outlier Factor', 'value': 'lof'},
                {'label': 'ISO-Forest', 'value': 'iso'},
                {'label': 'Eliptic Envelope', 'value': 'elip'},
            ],
            value='model'
        )
    ]),
    html.Br(),
    html.Div([
        html.Label('Data Set:'),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[
                {'label': 'BendODOT RTT', 'value': 'bendodot_router'},
                {'label': 'WagonTireTX', 'value': 'wagontire_router7_tx'},
                {'label': 'WagonTireRX', 'value': 'wagontire_router7_rx'},
                {'label': 'BryantRX', 'value': 'bryant_edge_rx'},
                {'label': 'BryantTX', 'value': 'bryant_edge_tx'},
                {'label': 'Blue Edge RTT', 'value': 'blue_edge_rtt'},
                {'label': 'Bryant Hog RTT', 'value': 'bryant_hog_rtt'},
                #{'label': 'Bryant Edge RTT', 'value': 'bryant_edge_rv55_rtt'}

            ],
            value='dataset'
        )
    ]),
    html.Br(),
    html.Button('Submit', id='submit-button'),
    html.Br(),
    html.Hr(),
    html.Br(),
    html.Br(),
    dcc.Graph(id='my-graph'),
    html.Br(),
    html.Hr(),
    html.Br(),
])

@dash.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('model-dropdown', 'value'),
     dash.dependencies.State('dataset-dropdown', 'value')]
)
def update_figure(n_clicks, model, dataset):
    if n_clicks is None:
        return dash.no_update
    
    dataframe, metadf = apimain(model, dataset)

    df = dataframe

    figure = px.line(df, x='timestamp', y='value')

    if model == 'multi':
        figure.add_trace(
            go.Scatter(
                x=df[df['anomaly_value'] > 2]['timestamp'],
                y=df[df['anomaly_value'] > 2]['value'],
                mode='markers',
                marker_symbol='circle',
                marker_size=15,
                name='Anomaly',
                #hoverinfo='all',
                hoverinfo='text+x+y',  # Add 'text' to include custom hover text
                text=df[df['anomaly_value'] > 2]['anomaly_value'].astype(str),  # Specify the text to be displayed on hover
                marker=dict(
                    color=df[df['anomaly_value'] > 2]['anomaly_value'],
                    colorscale=[[0, 'orange'], [0.5, 'red'], [1, 'black']],
                    showscale=True,
                )
            )
        )

    else:
        figure.add_trace(
            go.Scatter(
                x=df[df['anomaly_value'] > 0]['timestamp'],
                y=df[df['anomaly_value'] > 0]['value'],
                mode='markers',
                marker_symbol='circle',
                marker_size=15,
                name='Anomaly',
                hoverinfo='all',
            )
        )
    
    figure.update_traces(marker={'size': 8})

    figure.update_layout(
        autosize=False,
        title='Anomaly Detection Results',
        xaxis_title='Timestamp',
        yaxis_title='Value',
        width=1400,
        height=600,
        margin=dict(
            l=50,
            r=50,
            b=200,
            t=50,
            pad=4
        ),
        legend=dict(
            xanchor="left",
            x=0,
        )
    )

    con_val = metadf['con_val'].values[0]
    figure.add_annotation(dict(font=dict(color='Black',size=15),x=0,y=-0.2,showarrow=False,textangle=0,xanchor='left',xref="paper",yref="paper",
                               text="Meta Data: Contamination Value = " + str(con_val),
    ))
    figure.add_annotation(dict(font=dict(color='Black',size=15),x=0,y=-0.3,showarrow=False,textangle=0,xanchor='left',xref="paper",yref="paper",
                               text="F1 Score = ",
    ))

    return figure