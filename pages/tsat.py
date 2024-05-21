import dash
from dash import html, dcc, dash_table, Input, Output, State, callback
import pandas as pd
from DashDataAPI import apimain, apiupload
import plotly.express as px
import plotly.graph_objects as go
import os

dash.register_page(__name__, path='/tsat', title='Time Series Anomaly Detection Tool')

UPLOAD_DIRECTORY = "/useruploads"

layout = html.Div([
    html.H1('Anomaly Detection Tool'),

    html.Div([
        html.P("This tool allows you to analyze time series data for anomalies using Machine Learning techniques."),
        html.P("Select a model and an example dataset, then click submit generate Anomaly Detection results."),
        html.P("To analyze your own data for anomalies, click on the Select a Single File box and upload your desired dataset. After uploading the data, click on 'Analyze File Data' to generate the anomaly detection results."),
        html.P("The Anomaly Confidence Classifier Model classifies anomalies on a confidence scale out of 100, keep in mind a confidence score of 100 does not mean it has an accuracy of 100%.")
    ], style={'margin-bottom': '20px'}),
    html.Hr(),
    html.Div([
        html.Div([
            html.Label('Model:'),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Anomaly Confidence Classifier (Majority Vote)', 'value': 'multi'},
                    {'label': 'Discrete Anomaly Detection', 'value': 'gen'},
                ],
                value='model'
            )
        ], style={'width': '49%', 'display': 'inline-block'}),
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

                ],
                value='dataset'
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
    html.Br(),
    html.Div([
        dcc.Upload(id='upload-data_da',
                   children=html.Div(['Drag and Drop or ', html.A('Select a Single File')]),
                   style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                          'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                   # Do not allow multiple files to upload
                   multiple=False
                   ),
        html.Button('Upload File', id='upload_button_da'),
        html.Br(),
        html.Br(),
        html.Button('Analyze File Data', id='analyze_button_da', style={'display': 'none'}),
        html.Br(),
        html.Output(id='file_uploaded_da'),
    ]),
    html.Br(),
    html.Button('Submit', id='submit-button', style={'display': 'inline-block'}),
    html.Br(),
    html.Hr(),
    html.Br(),
    html.Br(),
    dcc.Graph(id='graph'),
    html.Br(),
    html.Hr(),
    html.Br(),
])

@callback(
    Output('file_uploaded_da', 'children'),
    Output('analyze_button_da', 'style'),
    Input('upload_button_da', 'n_clicks'),
    State('upload-data_da', 'contents'),
    State('upload-data_da', 'filename'),
    prevent_initial_call=True
)
def save_uploaded_file(n_clicks, contents, filename):
    if n_clicks:
        path = os.getcwd()
        if contents is not None:
            # Construct the full path to save the file
            filepath = path + UPLOAD_DIRECTORY + "/" + filename
            
            # Save the file to the upload directory
            with open(filepath, 'xb') as f:
                f.write(contents.encode('utf8'))
            
            return f"File '{filename}' uploaded successfully and saved.", {'display': 'block'}
        else:
            return "No file uploaded.", {'display': 'none'}
    else:
        return dash.no_update, {'display': 'none'}
    

@callback(
    Output('graph', 'figure'),
    Input('analyze_button_da', 'n_clicks'),
    Input('submit-button', 'n_clicks'),
    State('model-dropdown', 'value'),
    State('dataset-dropdown', 'value'),
    State('upload-data_da', 'filename'),
    prevent_initial_call=True
)

def update_figure(analyze_clicks, submit_clicks, model, dataset, filename):

    def gen_figure(dataframe, model, metadf):

        df = dataframe

        figure = px.line(df, x='timestamp', y='value')

        if model == 'multi':
            max_anomaly_score = 6  # Define the maximum anomaly score
            anomaly_confidence = df[df['anomaly_indices'] > 4]['anomaly_indices'] / max_anomaly_score * 100

            scatter_trace = go.Scatter(
                x=df[df['anomaly_indices'] > 4]['timestamp'],
                y=df[df['anomaly_indices'] > 4]['value'],
                mode='markers',
                marker_symbol='circle',
                marker_size=15,
                name='Anomaly',
                hoverinfo='text+x+y',  # Add 'text' to include custom hover text
                text=anomaly_confidence,  # Specify the confidence values for hover
                marker=dict(
                    color=anomaly_confidence,  # Use the confidence values as color
                    colorscale=[[0, 'red'], [1, 'black']],
                    showscale=True,
                )
            )

            figure.add_trace(scatter_trace)
            
            if len(scatter_trace.x) > 0:
                figure.add_annotation(dict(font=dict(color='Black',size=12),x=.97,y=1.056,showarrow=False,textangle=0,xanchor='left',xref="paper",yref="paper",
                                    text="Anomaly Confidence:",
                ))

        else:
            figure.add_trace(
                go.Scatter(
                    x=df[df['anomaly_indices'] > 0]['timestamp'],
                    y=df[df['anomaly_indices'] > 0]['value'],
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
            yaxis_title='RTT (ms)',
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
        #figure.add_annotation(dict(font=dict(color='Black',size=15),x=0,y=-0.3,showarrow=False,textangle=0,xanchor='left',xref="paper",yref="paper",
        #                        text="F1 Score = ",
        #))

        return figure

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'analyze_button_da':
        # Implement logic for analyzing the uploaded data and generating the figure
        # Return the figure
        if analyze_clicks:
            print(f"Analyzing file: {filename}")
            model = model

            dataframe, metadf = apiupload(model, filename)

            return gen_figure(dataframe, model, metadf)

        return dash.no_update

    elif triggered_id == 'submit-button':
        if submit_clicks:
            dataframe, metadf = apimain(model, dataset)

            return gen_figure(dataframe, model, metadf)
            
    else:
        return dash.no_update