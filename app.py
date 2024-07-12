import requests
import random
import socket
import dash
import numpy as np
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.io as pio
import plotly.graph_objects as go
from dash import no_update
import flask
import math 
import os 

pio.templates.default = "plotly_dark"

# Load the dataset
data = pd.read_json("Data/output.json")

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css"
FA = "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc_css])
server = app.server

curr_ip = '54.215.188.28'
app.layout = dbc.Container(
    children=[
        dash.page_container,
        dbc.Row(
            children=[
                # Column for user controls
                dbc.Col(
                    className="div-user-controls bg-dark",
                    children=[
                        html.H2("Delivering Insights with DoorDash"),
                        html.P(
                            """Select Market IDs and Days of the Week to filter the data.""",
                            style={'padding-left': '12px', 'padding-top': '15px'}
                        ),
                        html.Label("Market (City):", style={'padding-left': '12px', 'padding-top': '12px'}),
                        dcc.Dropdown(
                            id="market-id-dropdown",
                            options=[
                                {"label": market_id, "value": market_id}
                                for market_id in data["market_id"].unique()
                            ],
                            multi=True,
                            placeholder="Select City",
                        ),
                        html.Label("Day of Week:", style={'padding-left': '12px', 'padding-top': '15px'}),
                        dcc.Dropdown(
                            id="day-of-week-dropdown",
                            options=[
                                {"label": day, "value": day}
                                for day in data["day_of_week"].unique()
                            ],
                            multi=True,
                            placeholder="Select Day of Week"
                        ),
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    children=[
                                            dbc.Col(
                                                children=[
                                                    html.A(
                                                        dbc.Button("Visualize", id="visualize-button", className="custom-visualize-button"),
                                                        href = f'http://{curr_ip}:8002'
                                                    ),
                                                ],
                                                width="auto", className="text-center", style={'padding-right': '10%'}
                                            ),
                                    ],
                                    width="auto", className="text-center", style={'padding-right': '10%'}
                                ),
                                dbc.Col(
                                    children=[
                                            dbc.Col(
                                                children=[
                                                    html.A(
                                                        dbc.Button("Predict", id="predict-button", className="custom-predict-button mr-2"),
							href = f'http://{curr_ip}:8001'
                                                    ),
                                                ],
                                                width="auto", className="text-center", style={'padding-right': '10%'}
                                            ),
                                    ],
                                    width="auto", className="text-center", style={'padding-right': '10%'}
                                ),                                                          ],
                            justify="center",
                            style={'padding-top': '10%'}
                        ),
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    html.A(
                                        html.Img(
                                            className="fab fa-linkedin fa-2x",
                                            src=app.get_asset_url("LI_Logo.png"),
                                            style={'height': '20px', 'width': 'auto'}
                                        ),
                                        href="https://www.linkedin.com/in/chiraglakhanpal/",
                                    ),
                                    className="text-center ",
                                    style={'padding-right': '20px', 'border-right': '1px solid yellow'}
                                ),
                                dbc.Col(
                                    html.A(
                                        html.Img(
                                            className="logo",
                                            src=app.get_asset_url("GitHub_Logo.png"),
                                            style={'height': '20px', 'width': 'auto'}
                                        ),
                                        href="https://github.com/ChiragLakhanpal",
                                    ),
                                    className="text-center ",
                                    style={'padding-right': '20px'}
                                ),
                            ],
                            className='logos align-items-center text-center'
                        ),
                    ],
                    width=4
                ),
                # Column for app graphs and plots
                dbc.Col(
                    className="div-for-charts bg-dark",
                    children=[
                        dbc.Row(
                            children=[
                                dcc.Graph(id="graph1", config={'displayModeBar': False}),
                            ],
                        ),
                        dbc.Row(
                            children=[
                                dcc.Graph(id="eta-histogram", config={'displayModeBar': False}),
                            ]
                        )
                    ], width=8
                ),
            ],
        ),
    ],
    fluid=True,
    style={"padding": "0px 20px 0px 20px"},
)


################### Map Plot Configuration ###################

city_coordinates = {
"Chicago": {"lat": 41.8781136, "lng": -87.6297982},
"Phoenix": {"lat": 33.4483771, "lng": -112.0740373},
"Philadelphia": {"lat": 39.9525839, "lng": -75.1652215},
"New York": {"lat": 40.7127753, "lng": -74.0059728},
"Houston": {"lat": 29.7604267, "lng": -95.3698028},
"Los Angeles": {"lat": 34.0522342, "lng": -118.2436849},
}

data["lat"] = data["market_id"].apply(lambda x: city_coordinates.get(x, {"lat": None, "lng": None})["lat"])
data["lng"] = data["market_id"].apply(lambda x: city_coordinates.get(x, {"lat": None, "lng": None})["lng"])

mapbox_token = "pk.eyJ1IjoiY2hpcmFnbGFraGFucGFsIiwiYSI6ImNsZ2FkMGE1czAzMDczaG56djYwNjFwcTIifQ.ReeYivk9-ijgG-E0vxPTRw"
px.set_mapbox_access_token(mapbox_token)

def update_map(market_ids, days_of_week):
    filtered_data = data.copy()

    if market_ids:
        filtered_data = filtered_data[filtered_data["market_id"].isin(market_ids)]

    if days_of_week:
        filtered_data = filtered_data[filtered_data["day_of_week"].isin(days_of_week)]

    heatmap_data = filtered_data.groupby(['lat', 'lng', 'market_id']).size().reset_index(name='num_orders')

    scatter_map = px.scatter_mapbox(
        heatmap_data,
        lat="lat",
        lon="lng",
        color="num_orders",
        color_continuous_scale=px.colors.sequential.Agsunset,
        size="num_orders",
        hover_name="market_id",
        zoom=3,
        color_continuous_midpoint = np.average(heatmap_data['num_orders'], weights=heatmap_data['num_orders'])
    )

    scatter_map.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        mapbox=dict(bearing=0, pitch=0, zoom=3),
        scene=dict(aspectmode='auto'),
        coloraxis=dict(
            colorbar=dict(
                title=dict(text="Number of Orders", font=dict(size=14)),
                tickfont=dict(size=12),
                len=0.5,
                thickness=15,
                x=0.87,
                y=0.4
            )
        )
    )
    return scatter_map

################### Map Plot ###################

@app.callback(
Output("graph1", "figure"),
Input("market-id-dropdown", "value"),
Input("day-of-week-dropdown", "value"),
)

def update_graph1(market_ids, days_of_week):
    scatter_map = update_map(market_ids, days_of_week)

    return scatter_map

################### Histogram ###################

@app.callback(
Output("eta-histogram", "figure"),
Input("market-id-dropdown", "value"),
Input("day-of-week-dropdown", "value"),
)


def update_histogram(market_ids, days_of_week):
    filtered_data = data.copy()

    if market_ids:  
        filtered_data = filtered_data[filtered_data["market_id"].isin(market_ids)]

    if days_of_week:
        filtered_data = filtered_data[filtered_data["day_of_week"].isin(days_of_week)]

    hourly_counts = filtered_data.created_at.dt.hour.value_counts().sort_index()

    max_y = int(math.ceil(max(count for hour, count in hourly_counts.items()) / 5000.0)) * 5000
    y_tick_step = 5000

    fig = go.Figure()

    for idx, count in hourly_counts.items():
        fig.add_trace(go.Bar(
            x=[idx],
            y=[count],
            marker_color=[f"hsl({360 * idx / 24}, 50%, 50%)"],
            name=str(idx),
            text=[count],
            textposition='outside',
            showlegend=False,
            textfont=dict(size=10),
        ))

    hour_labels = [f"{hour:02d}:00" if hour % 2 == 0 else '' for hour in range(24)]
    
    fig.update_yaxes(title_text='Orders', color="#FFD500")
    fig.update_xaxes(title_text='Hour', color="#FFD500")

    fig.update_layout(
        title = {
        'text': 'Hourly Orders',
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font_family': 'Avenir',
        'font_size': 20},
        bargap=0.01,
        title_font_color = '#FFD500',
        xaxis=dict(tickmode='array', tickvals=list(range(24)), ticktext=hour_labels),
        yaxis=dict(range=[0, max_y], tick0=0, dtick=y_tick_step)
    )

    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug = True, port= 8000)
