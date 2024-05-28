import dash
from dash import Dash, html, dcc
import os
import atexit

PORT = 5080
ADDRESS = '127.0.0.1'

app = Dash(__name__, use_pages=True)

# Folder to clear upon shutdown
folder_to_clear = '/useruploads'

def clear_folder():
    """Function to clear the contents of a specific folder."""
    filepath = os.getcwd() + folder_to_clear
    if os.path.exists(filepath):
        for filename in os.listdir(filepath):
            file = os.path.join(filepath, filename)
            try:
                os.remove(file)
            except Exception as e:
                print(f"Failed to delete {file}. Reason: {e}")
        print(f"Contents of {folder_to_clear} cleared successfully.")
    else:
        print(f"Folder {folder_to_clear} does not exist.")

# Registering the clear_folder function to be called upon exit
atexit.register(clear_folder)

navbar = html.Div(
    [
        html.H2("Time Series Anomaly Detetion"),
        html.Hr(),
        html.Div([
            dcc.Link('Home', href='/', style={'background-color': 'gray', 'color': 'black', 'padding': '5px 10px', 'margin': '5px 10px', 'border-radius': '5px', 'font-size': '120%', 'transform': 'scale(1.2)', 'text-decoration': 'none'}),
            dcc.Link('Time Series Anomaly Tool', href='/tsat', style={'background-color': 'gray', 'color': 'black', 'padding': '5px 10px', 'margin': '5px 30px', 'border-radius': '5px', 'font-size': '120%', 'transform': 'scale(1.2)', 'text-decoration': 'none'}),
            dcc.Link('Anomaly Detection Testing Menu', href='/adm', style={'background-color': 'gray', 'color': 'black', 'padding': '5px 10px', 'margin': '5px 30px', 'border-radius': '5px', 'font-size': '120%', 'transform': 'scale(1.2)', 'text-decoration': 'none'}),
            dcc.Link('About', href='/about', style={'background-color': 'gray', 'color': 'black', 'padding': '5px 10px', 'margin': '5px 20px', 'border-radius': '5px', 'font-size': '120%', 'transform': 'scale(1.2)', 'text-decoration': 'none'}),
        ], style={'display': 'flex', 'flex-direction': 'row'}),
        html.Br(),
        html.Hr(),
    ],

)

app.layout = html.Div(
    [
        dcc.Location(id="url"), 
        navbar,
        html.Div(
            [
                dash.page_container,
            ],
            className="content",
            id="page-content",
        ),
    ]
)

if __name__ == '__main__':
    app.run_server(
        port=PORT,
        host=ADDRESS,
        debug=True
    )