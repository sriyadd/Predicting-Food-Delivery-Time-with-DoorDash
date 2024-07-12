import random
import dash
import numpy as np
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.io as pio
import plotly.graph_objects as go
import scipy.stats as stats
import plotly.figure_factory as ff
from dash import no_update
from sklearn.neural_network import MLPRegressor
import joblib
import datetime
from sklearn.preprocessing import OneHotEncoder
import os
import base64

pio.templates.default = "plotly_dark"
model = joblib.load('Model/model.pkl')

# Load the dataset
data = pd.read_json("Data/output.json")
model_data = pd.read_json("Data/model_data.json")

dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc_css])


# Set the file path and name of your report PDF
report_file = 'Report/DoorDash_Report_by_Chirag_Lakhanpal.pdf'

# Define a function to encode the file
def encode_file(file):
    with open(file, 'rb') as f:
        file_bytes = f.read()
    encoded = base64.b64encode(file_bytes).decode()
    return encoded


app.layout = dbc.Container(
    children=[
        dbc.Row(
            children=[
                dbc.Col(
                    dbc.Card(
                        children=[
                            dbc.CardHeader('Control Panel'),
                            dbc.CardBody(
                                children=[
                                dbc.ListGroup(
                                    children=[
                                        dbc.ListGroupItem([
                                            html.H6('Ordered on', style={'margin-right': '5%', 'padding-top': '5%'}),
                                            dcc.DatePickerSingle(
                                                id="date-picker-dropdown",
                                                min_date_allowed=pd.to_datetime('2015-01-01'),
                                                max_date_allowed=pd.to_datetime('2030-12-31'),
                                                initial_visible_month=pd.to_datetime('2022-01-01'),
                                                date=None,
                                                display_format='MMMM DD, YYYY',
                                                className='date-picker',
                                                placeholder='Select a Date'
                                            ),
                                            html.H6('at', style={'margin-right': '5%','margin-left': '4%', 'padding-top': '5%'}),
                                            dcc.Dropdown(
                                                id='hour-dropdown',
                                                options=[{'label': f'{i:02d}', 'value': i} for i in range(24)],
                                                placeholder='Hour',
                                                clearable=False,
                                                style={"color": "#FFD500",
                                                        "font-size": "14px",
                                                        "height": "30px",
                                                        "width": "100px",
                                                        "background-color": "#303030",
                                                        "margin-right": '5px',
                                                        'margin-bottom': '5px'}
                                            ),
                                            dcc.Dropdown(
                                                id='minute-dropdown',
                                                options=[{'label': f'{i:02d}', 'value': i} for i in range(0, 60, 1)],
                                                placeholder='Minute',
                                                clearable=False,
                                                style={"color": "#FFD500",
                                                        "font-size": "14px",
                                                        "height": "30px",
                                                        "width": "100px",
                                                        "background-color": "#303030",
                                                        'margin-bottom': '5px'}
                                            ),
                                        ], style={'display': 'flex', 'align-items': 'center','border-bottom': '0 '}),                                        
                                        dbc.ListGroupItem(
                                            children=[
                                                html.H6('Payment Type', style={'margin-right': '10%', 'padding-top': '5%'}),
                                                html.Div(
                                                    dcc.RadioItems(
                                                        id='payment-type-radioitems',
                                                        options=[{'label': 'Online', 'value': 'Online'},
                                                                {'label': 'Cash', 'value': 'Cash'}],
                                                        labelStyle={'margin-right': '40px', 'font-size': '14px', 'color': '#FFD500', 'margin-left': '10px'},
                                                        inline=True,
                                                        className='payment-type-radioitems yellow-radio',  # Added the 'yellow-radio' class
                                                    ),
                                                    style={'width': '100%', 'display': 'flex', 'justify-content': 'center', 'align-items': 'flex-start'}
                                                )
                                            ], style={'display': 'flex', 'align-items': 'center', 'border-bottom': '0'}
                                        ),
                                        dbc.ListGroupItem(
                                            children=[
                                                html.H6('Delivery Distance', style={'margin-right': '10%', 'padding-top': '5%'}),
                                                dcc.Slider(
                                                    id='delivery-distance-slider',
                                                    min=1,
                                                    max=21,
                                                    step=0.1,
                                                    value=5,
                                                    marks={i: f'{i}' for i in range(1, 22, 2)},
                                                    className='delivery-distance-slider',
                                                    tooltip={'always_visible': False, 'placement': 'top'}
                                                )
                                            ], style={'display': 'flex', 'align-items': 'center', 'border-bottom': '0 '}
                                        ),
                                        dbc.ListGroupItem(
                                            children=[
                                                html.H6('City (Market)', style={'margin-right': '5%', 'padding-top': '5%'}),
                                                html.Div(
                                                    dcc.Dropdown(
                                                        id='Market-dropdown',
                                                        options=[{'label': city, 'value': city} for city in data.market_id.unique()],
                                                        placeholder='Select the City',
                                                        clearable=False,
                                                        style={"color": "#FFD500",
                                                            "font-size": "14px",
                                                            "height": "30px",
                                                            "width": "200px",
                                                            "background-color": "#303030",
                                                            "margin-right": '5px'}
                                                    ),
                                                    style={'width': '100%', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}
                                                )
                                            ], style={'display': 'flex', 'align-items': 'center', 'border-bottom': '0'}
                                        ),
                                        dbc.ListGroupItem(
                                            children=[
                                                html.H6('Cuisine', style={'margin-left': '3%','margin-right': '5%', 'padding-top': '5%'}),
                                                html.Div(
                                                    dcc.Dropdown(
                                                        id='Cuisine-dropdown',
                                                        options=[{'label': city, 'value': city} for city in data.store_primary_category.unique()],
                                                        placeholder='Select the Cuisine',
                                                        clearable=False,
                                                        style={"color": "#FFD500",
                                                            "font-size": "14px",
                                                            "height": "30px",
                                                            "width": "200px",  
                                                            "background-color": "#303030",
                                                            "margin-right": '5px'}
                                                    ),
                                                    style={'width': '100%', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center','margin-left': '6%','margin-right': '5%'}
                                                )
                                            ], style={'display': 'flex', 'align-items': 'center', 'border-bottom': '0'}
                                        ),
                                        dbc.ListGroupItem(
                                            children=[
                                                html.H6('Items', style={'margin-left': '3%', 'margin-right': '5%', 'padding-top': '5%'}),
                                                html.Div(
                                                    dbc.Input(
                                                        id='number-of-items-input',
                                                        type='number',
                                                        min=0,
                                                        max=15,
                                                        step=1,
                                                        placeholder='0',
                                                        className='yellow-placeholder',
                                                        style={
                                                            "color": "#FFD500",
                                                            "font-size": "14px",
                                                            "height": "40px",
                                                            "width": "200px",
                                                            "background-color": "#303030",
                                                            "margin-right": '5px',
                                                            "border": "1px solid #FFFFFF",
                                                            'justify-content': 'center',
                                                            "text-align": "center"
                                                        }
                                                    ),
                                                    style={'width': '100%', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'margin-left': '8%', 'margin-right': '5%'}
                                                )
                                            ], style={'display': 'flex', 'align-items': 'center', 'border-bottom': '0'}
                                        ),
                                        dbc.ListGroupItem(
                                            children=[
                                                dbc.Button(
                                                    "Where's My Grub?",
                                                    id='predict-button',
                                                    color='warning',
                                                    size='lg',
                                                    style={
                                                        'font-size': '14px',
                                                        'font-weight': 'bold',
                                                        'margin-top': '10px',
                                                        'background-color': '#FFD500',
                                                        'border-color': 'None',
                                                        'color': '#303030',
                                                        'margin-right': '10%', 
                                                        'margin-left': '5%',
                                                    }
                                                ),
                                            html.A(
                                                    'Want to Learn More?',
                                                    id='download-button',
                                                    download=report_file,
                                                    href=f'data:application/pdf;base64,{encode_file(report_file)}',
                                                    target="_blank",
                                                    style={
                                                        'font-size': '14px',
                                                        'font-weight': 'bold',
                                                        'margin-top': '10px',
                                                        'background-color': '#FFD500',
                                                        'border-color': 'None',
                                                        'color': '#303030',
                                                        'padding': '10px',
                                                        'border-radius': '5px',
                                                        'text-decoration': 'none',
                                                        'margin-left': '5%',
                                                        'margin-right': '5%' 
                                                    }
                                                )
                                            ],
                                            style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'border-bottom': '0'}
                                        )

                                        ], flush=True

                                    ),
                                   dbc.Col(
                                        dbc.Card(
                                            children=[
                                                dbc.CardBody(children=[
                                                    html.Div(
                                                        id='prediction-result',
                                                        style={'color': '#FFD500', 'font-size': '16px', 'padding': '10px'}
                                                    )
                                                ])
                                            ],
                                            className='order-details',
                                            style={'border': '0'}
                                        ), width={'size': 5, 'offset': 1, 'order': 1, 'lg': 12}
                                    )                                                                                
                                ]
                            ),                          
                        ],
                        className='control-panel'
                    ), width={'size': 12, 'offset': 0, 'order': 1, 'lg': 5}
                
                )
            ]
        )
    ]
)

def preprocess_data(input_data, model_data):

    input_data_processed = input_data.copy()
    
    # Scale numerical columns
    numerical_columns = input_data_processed.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        input_data_processed[col] = (input_data_processed[col] - model_data[col].mean()) / model_data[col].std()

    return input_data_processed

selected_protocol = random.choice(range(1, 8))

@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('date-picker-dropdown', 'date'),
     State('hour-dropdown', 'value'),
     State('minute-dropdown', 'value'),
     State('payment-type-radioitems', 'value'),
     State('delivery-distance-slider', 'value'),
     State('Market-dropdown', 'value'),
     State('Cuisine-dropdown', 'value'),
     State('number-of-items-input', 'value')]
)
def update_predict_delivery_time(n_clicks, date, hour, minute, payment_type, delivery_distance, market, cuisine, num_items):

    date = pd.to_datetime(date)

    if n_clicks is None:
        return html.H3("Input your grub info, give that button a click, and watch your ETA magically appear!")
    if date is None or hour is None or minute is None or payment_type is None or delivery_distance is None or market is None or cuisine is None or num_items is None:
        return html.H3("Make sure to complete all the fields so that we can calculate your tasty ETA!")


    # Prepare the input data
    input_data = pd.DataFrame(data=[{
        'total_items': num_items,
        'subtotal': model_data['subtotal'].mean(),
        'num_distinct_items': num_items,
        'min_item_price': model_data['min_item_price'].mean(),  
        'max_item_price': model_data['max_item_price'].mean(),  
        'total_onshift_dashers': model_data['total_onshift_dashers'].mean(),
        'total_busy_dashers': model_data['total_busy_dashers'].mean(),
        'total_outstanding_orders': model_data['total_outstanding_orders'].mean(),
        'estimated_order_place_duration': model_data['estimated_order_place_duration'].mean(),
        'estimated_store_to_consumer_driving_duration': model_data['estimated_store_to_consumer_driving_duration'].mean(),
        'market_id_Chicago': 1 if market == 'Chicago' else 0,
        'market_id_Houston': 1 if market == 'Houston' else 0,
        'market_id_Los Angeles': 1 if market == 'Los Angeles' else 0,
        'market_id_New York': 1 if market == 'New York' else 0,
        'market_id_Philadelphia': 1 if market == 'Philadelphia' else 0,
        'market_id_Phoenix': 1 if market == 'Phoenix' else 0,        
        'order_protocol_1': 1 if selected_protocol == 1 else 0,
        'order_protocol_2': 1 if selected_protocol == 2 else 0,
        'order_protocol_3': 1 if selected_protocol == 3 else 0,
        'order_protocol_4': 1 if selected_protocol == 4 else 0,
        'order_protocol_5': 1 if selected_protocol == 5 else 0,
        'order_protocol_6': 1 if selected_protocol == 6 else 0,
        'order_protocol_7': 1 if selected_protocol == 7 else 0,
        'store_primary_category_afghan': 1 if cuisine == 'afghan' else 0,
        'store_primary_category_african': 1 if cuisine == 'african' else 0,
        'store_primary_category_alcohol': 1 if cuisine == 'alcohol' else 0,
        'store_primary_category_alcohol-plus-food': 1 if cuisine == 'alcohol-plus-food' else 0,
        'store_primary_category_american': 1 if cuisine == 'american' else 0,
        'store_primary_category_argentine': 1 if cuisine == 'argentine' else 0,
        'store_primary_category_asian': 1 if cuisine == 'asian' else 0,
        'store_primary_category_barbecue': 1 if cuisine == 'barbecue' else 0,
        'store_primary_category_belgian': 1 if cuisine == 'belgian' else 0,
        'store_primary_category_brazilian': 1 if cuisine == 'brazilian' else 0,
        'store_primary_category_breakfast': 1 if cuisine == 'breakfast' else 0,
        'store_primary_category_british': 1 if cuisine == 'british' else 0,
        'store_primary_category_bubble-tea': 1 if cuisine == 'bubble-tea' else 0,
        'store_primary_category_burger': 1 if cuisine == 'burger' else 0,
        'store_primary_category_burmese': 1 if cuisine == 'burmese' else 0,
        'store_primary_category_cafe': 1 if cuisine == 'cafe' else 0,
        'store_primary_category_cajun': 1 if cuisine == 'cajun' else 0,
        'store_primary_category_caribbean': 1 if cuisine == 'caribbean' else 0,
        'store_primary_category_catering': 1 if cuisine == 'catering' else 0,
        'store_primary_category_cheese': 1 if cuisine == 'cheese' else 0,
        'store_primary_category_chinese': 1 if cuisine == 'chinese' else 0,
        'store_primary_category_chocolate': 1 if cuisine == 'chocolate' else 0,
        'store_primary_category_comfort-food': 1 if cuisine == 'comfort-food' else 0,
        'store_primary_category_convenience-store' : 1 if cuisine == 'convenience-store' else 0,
        'store_primary_category_dessert': 1 if cuisine == 'dessert' else 0,
        'store_primary_category_dim-sum': 1 if cuisine == 'dim-sum' else 0,
        'store_primary_category_ethiopian': 1 if cuisine == 'ethiopian' else 0,
        'store_primary_category_european': 1 if cuisine == 'european' else 0,
        'store_primary_category_fast': 1 if cuisine == 'fast' else 0,
        'store_primary_category_filipino': 1 if cuisine == 'filipino' else 0,
        'store_primary_category_french': 1 if cuisine == 'french' else 0,
        'store_primary_category_gastropub': 1 if cuisine == 'gastropub' else 0,
        'store_primary_category_german': 1 if cuisine == 'german' else 0,
        'store_primary_category_gluten-free': 1 if cuisine == 'gluten-free' else 0,
        'store_primary_category_greek': 1 if cuisine == 'greek' else 0,
        'store_primary_category_hawaiian': 1 if cuisine == 'hawaiian' else 0,
        'store_primary_category_indian': 1 if cuisine == 'indian' else 0,
        'store_primary_category_indonesian': 1 if cuisine == 'indonesian' else 0,
        'store_primary_category_irish': 1 if cuisine == 'irish' else 0,
        'store_primary_category_italian': 1 if cuisine == 'italian' else 0,
        'store_primary_category_japanese': 1 if cuisine == 'japanese' else 0,
        'store_primary_category_korean': 1 if cuisine == 'korean' else 0,
        'store_primary_category_kosher': 1 if cuisine == 'kosher' else 0,
        'store_primary_category_latin-american': 1 if cuisine == 'latin-american' else 0,
        'store_primary_category_lebanese': 1 if cuisine == 'lebanese' else 0,
        'store_primary_category_malaysian': 1 if cuisine == 'malaysian' else 0,
        'store_primary_category_mediterranean': 1 if cuisine == 'mediterranean' else 0,
        'store_primary_category_mexican': 1 if cuisine == 'mexican' else 0,
        'store_primary_category_middle-eastern': 1 if cuisine == 'middle-eastern' else 0,
        'store_primary_category_moroccan': 1 if cuisine == 'moroccan' else 0,
        'store_primary_category_nepalese': 1 if cuisine == 'nepalese' else 0,
        'store_primary_category_other': 1 if cuisine == 'other' else 0,
        'store_primary_category_pakistani': 1 if cuisine == 'pakistani' else 0,
        'store_primary_category_pasta': 1 if cuisine == 'pasta' else 0,
        'store_primary_category_persian': 1 if cuisine == 'persian' else 0,
        'store_primary_category_peruvian': 1 if cuisine == 'peruvian' else 0,
        'store_primary_category_pizza': 1 if cuisine == 'pizza' else 0,
        'store_primary_category_russian': 1 if cuisine == 'russian' else 0,
        'store_primary_category_salad': 1 if cuisine == 'salad' else 0,
        'store_primary_category_sandwich': 1 if cuisine == 'sandwich' else 0,
        'store_primary_category_seafood': 1 if cuisine == 'seafood' else 0,
        'store_primary_category_singaporean': 1 if cuisine == 'singaporean' else 0,
        'store_primary_category_smoothie': 1 if cuisine == 'smoothie' else 0,
        'store_primary_category_soup': 1 if cuisine == 'soup' else 0,
        'store_primary_category_southern': 1 if cuisine == 'southern' else 0,
        'store_primary_category_spanish': 1 if cuisine == 'spanish' else 0,
        'store_primary_category_steak': 1 if cuisine == 'steak' else 0,
        'store_primary_category_sushi': 1 if cuisine == 'sushi' else 0,
        'store_primary_category_tapas': 1 if cuisine == 'tapas' else 0,
        'store_primary_category_thai': 1 if cuisine == 'thai' else 0,
        'store_primary_category_turkish': 1 if cuisine == 'turkish' else 0,
        'store_primary_category_vegan': 1 if cuisine == 'vegan' else 0,
        'store_primary_category_vegetarian': 1 if cuisine == 'vegetarian' else 0,
        'store_primary_category_vietnamese': 1 if cuisine == 'vietnamese' else 0,
        'created_at_year': date.year,
        'created_at_month': date.month,
        'created_at_day': date.day,
        'created_at_hour': hour,
        'created_at_minute': minute,
        'created_at_second': 0,
        'busy_rider_ratio': model_data['busy_rider_ratio'].mean(),
        'non_prep_duration': model_data['non_prep_duration'].mean(),
        'avg_price': model_data['avg_price'].mean(),
        'delivery_distance': delivery_distance,
        'order_density': model_data['order_density'].mean(),
        'day_of_week_Friday': 1 if date.weekday() == 4 else 0,
        'day_of_week_Monday': 1 if date.weekday() == 0 else 0,
        'day_of_week_Saturday': 1 if date.weekday() == 5 else 0,
        'day_of_week_Sunday': 1 if date.weekday() == 6 else 0,
        'day_of_week_Thursday': 1 if date.weekday() == 3 else 0,
        'day_of_week_Tuesday': 1 if date.weekday() == 1 else 0,
        'day_of_week_Wednesday': 1 if date.weekday() == 2 else 0,
    }])

# total_items subtotal num_distinct_items min_item_price max_item_price total_onshift_dashers total_busy_dashers total_outstanding_orders estimated_order_place_duration estimated_store_to_consumer_driving_duration market_id_Chicago market_id_Houston market_id_Los Angeles market_id_New York market_id_Philadelphia market_id_Phoenix order_protocol_1 order_protocol_2 order_protocol_3 order_protocol_4 order_protocol_5 order_protocol_6 order_protocol_7 store_primary_category_afghan store_primary_category_african store_primary_category_alcohol store_primary_category_alcohol-plus-food store_primary_category_american store_primary_category_argentine store_primary_category_asian store_primary_category_barbecue store_primary_category_belgian store_primary_category_brazilian store_primary_category_breakfast store_primary_category_british store_primary_category_bubble-tea store_primary_category_burger store_primary_category_burmese store_primary_category_cafe store_primary_category_cajun store_primary_category_caribbean store_primary_category_catering store_primary_category_cheese store_primary_category_chinese store_primary_category_chocolate store_primary_category_comfort-food store_primary_category_convenience-store store_primary_category_dessert store_primary_category_dim-sum store_primary_category_ethiopian store_primary_category_european store_primary_category_fast store_primary_category_filipino store_primary_category_french store_primary_category_gastropub store_primary_category_german store_primary_category_gluten-free store_primary_category_greek store_primary_category_hawaiian store_primary_category_indian store_primary_category_indonesian store_primary_category_irish store_primary_category_italian store_primary_category_japanese store_primary_category_korean store_primary_category_kosher store_primary_category_latin-american store_primary_category_lebanese store_primary_category_malaysian store_primary_category_mediterranean store_primary_category_mexican store_primary_category_middle-eastern store_primary_category_moroccan store_primary_category_nepalese store_primary_category_other store_primary_category_pakistani store_primary_category_pasta store_primary_category_persian store_primary_category_peruvian store_primary_category_pizza store_primary_category_russian store_primary_category_salad store_primary_category_sandwich store_primary_category_seafood store_primary_category_singaporean store_primary_category_smoothie store_primary_category_soup store_primary_category_southern store_primary_category_spanish store_primary_category_steak store_primary_category_sushi store_primary_category_tapas store_primary_category_thai store_primary_category_turkish store_primary_category_vegan store_primary_category_vegetarian store_primary_category_vietnamese created_at_year created_at_month created_at_day created_at_hour created_at_minute created_at_second busy_rider_ratio non_prep_duration avg_price delivery_distance order_density day_of_week_Friday day_of_week_Monday day_of_week_Saturday day_of_week_Sunday day_of_week_Thursday day_of_week_Tuesday day_of_week_Wednesday

    # Preprocess the input data
    preprocessed_data = preprocess_data(input_data, model_data)

    # Make a prediction
    eta = model.predict(preprocessed_data)
    
    # Convert the ETA to minutes
    eta_minutes = int(eta[0]//100)

    # Return the result
    return html.H3(f"Hang tight, your mouthwatering meal will be at your doorstep in around {eta_minutes} minutes!")

if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0',port=8001)


