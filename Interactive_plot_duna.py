#Copyright © 2025 INNOMOTICS 

import plotly.graph_objects as go
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
from flask import current_app, has_app_context
from config import timestamp, resample_dataframe

# Function to create Dash app
def create_dash_app(flask_server):
    dash_app = dash.Dash(
        __name__,
        server=flask_server,  # Attach Dash to Flask
        url_base_pathname="/simulator/",
        suppress_callback_exceptions=True
    )

    # Function to get the DataFrame dynamically with context check
    def get_df():
        if has_app_context():
            df = current_app.config.get('df_fingerprint', pd.DataFrame()).copy()
        else:
            return pd.DataFrame()  # Return empty DataFrame if context is missing

        if not df.empty:
            df[timestamp] = pd.to_datetime(df[timestamp], errors='coerce')
            df = df.set_index(timestamp).resample(resample_dataframe).first().reset_index()
            df = df.iloc[:5000]  # Limit rows efficiently

        return df

    # Columns to exclude
    exclude_columns = ['timestamp']
    include_columns = ['K5770AC WGF Fan actual speed [rpm]',
       'K5770AC WGF Fan actual current [A]',
       'K5026 TE1 Kiln temperature (Pyrometer 1)',
       'K5026 TE2 Kiln temperature (Pyrometer 2)',
       'Stone per cycle setpoint [kg]']
    
    # Function to generate dimensions dynamically
    def generate_dimensions(df):
        columns_to_plot = [col for col in df.columns if col not in exclude_columns and col in include_columns]
        return [
            dict(
                range=[df[col].min(), df[col].max()],
                label=col,
                values=df[col]
            ) for col in columns_to_plot
        ]

    # Create the initial plot
    df = get_df()
    fig = go.Figure(data=go.Parcoords(
        line=dict(color='#DFFF00'),
        dimensions=generate_dimensions(df) if not df.empty else [],
        unselected=dict(line=dict(color='white', opacity=0))
    ))

    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        annotations=[
            dict(
                text="Simulator",
                x=0.5,
                y=1.25,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=28, color='#DFFF00')
            )
        ],
        font=dict(color='#DFFF00', size=8),
        margin=dict(t=140, l=40, r=60, b=50)
    )

    # Dash Layout
    dash_app.layout = html.Div([
        html.Div([
            dcc.Graph(
                figure=fig,
                id='parallel-coordinates-plot',
                style={'width': '100%', 'height': '90vh'}
            )
        ], style={'padding': '20px', 'background-color': '#F0F0F0', 'width': '100%', 'height': 'auto', 'margin': '0'}),

        html.Div(id='dummy-div', style={'display': 'none'}),
    ], style={'backgroundColor': '#FFFD70', 'width': '100%', 'height': '100vh', 'margin': '0', 'overflow': 'hidden'})

    # Callback to update plot dynamically
    @dash_app.callback(
        Output('parallel-coordinates-plot', 'figure'),
        [Input('dummy-div', 'children')]
    )
    def update_figure(dummy):
        df = get_df()
        dimensions = generate_dimensions(df) if not df.empty else []

        fig = go.Figure(data=go.Parcoords(
            line=dict(color='#DFFF00'),
            dimensions=dimensions,
            unselected=dict(line=dict(color='white', opacity=0))
        ))

        fig.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            annotations=[
                dict(
                    text="Simulator",
                    x=0.5,
                    y=1.25,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    font=dict(size=28, color='#DFFF00')
                )
            ],
            font=dict(color='#DFFF00', size=8),
            margin=dict(t=140, l=40, r=60, b=50)
        )

        return fig

    return dash_app


