import random

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
import scipy.stats as stats
import plotly.figure_factory as ff


pio.templates.default = "plotly_dark"


# Load the dataset
data = pd.read_json("Data/output.json")

dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc_css])

dash.register_page(__name__)

app.layout = html.Div(
    [
        # Quick facts row
        html.Div(id='dummy-input', style={'display': 'none'}),
        dbc.Row(
            children=[
                dbc.Col(dbc.Card(className='card-quick-facts', children=[dbc.CardHeader('Order Galore: Total Count'), dbc.CardBody(id='total-orders-card')]), width=3),
                dbc.Col(dbc.Card(className='card-quick-facts', children=[dbc.CardHeader('Size Matters: Average Order'), dbc.CardBody(id='average-order-size-card')]), width=3),
                dbc.Col(dbc.Card(className='top-categories-card', children=[dbc.CardHeader('Fabulous Five: Top Categories'), dcc.Graph(id='top-category-bubble-chart', config={'displayModeBar': False})]), width=3),
                dbc.Col(dbc.Card(className='card-quick-facts', children=[dbc.CardHeader('Average Order Value'), dbc.CardBody(id='average-order-value-card')]), width=3),
            ],
            justify='center'
        ),

        # Tabs
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Overview & Market Insights",
                    children=[
                        # Line graph row
                        dbc.Row(
                            children=[
                                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='line-graph-city', config={'displayModeBar': False})), className="my-line-card"), width=12)
                            ],
                            justify='center'
                        ),

                        # Three graphs row
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody([
                                            dcc.Dropdown(id='numeric-columns-dropdown', options=[{'label': col, 'value': col} for col in data.select_dtypes(['int64', 'float64']).columns], value='eta',style={'borderColor': '#FFD500'}),
                                            dcc.Graph(id='all-feature-histogram', config={'displayModeBar': False})
                                        ]),
                                        className='my-histogram-card'
                                    ),
                                    width=4
                                ),
                                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='pie-chart', config= {'displayModeBar': False})), className='pie-chart'), width=4),
                                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='count-day-time', config={'displayModeBar': False})),  className="my-stack-card"), width=4)
                            ],
                            justify='center'
                        ),

                        # Two graphs row
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    children=[
                                                        dbc.Col(dcc.Dropdown(id='first-feature', options=[feature for feature in data.select_dtypes(['int64', 'float64']).columns],value='avg_price',style={'borderColor': '#FFD500'}), width=5),
                                                        dbc.Col(html.H6("vs", className='text-center', style={'margin-top': '6px'}), width=2),
                                                        dbc.Col(dcc.Dropdown(id='second-feature', options=[feature for feature in data.select_dtypes(['int64', 'float64']).columns],value='eta',style={'borderColor': '#FFD500'}), width=5)
                                                    ],
                                                    justify='center'
                                                ),
                                                dcc.Graph(id='all-scatter-plot', config={'displayModeBar': False})
                                            ]
                                        ), className='all-scatter-plot'
                                    ), width=6
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [dcc.Slider(id='select-top-cat', min=1, max=74, step=1, value=10, marks={i: f"{i}" for i in range(1, 75, 10)}, className='my-slider', updatemode='drag'),
                                            dcc.Graph(id='cat-bubble-plot', config={'displayModeBar': False})]
                                            ),
                                        className='my-buble-chart'
                                    ),
                                    width=6
                                )
                            ],
                            justify='center'
                        ),
                    ], tab_style={ 'width': '50%', 'text-align': 'center','font-family': 'Avenir','font-size': 20}, 
                       label_style={"color": "#FFD500"},active_tab_style={'border-top': '3px solid #FFD500'}
                                   
                ), 
                dbc.Tab(
                    label="Advanced Technical Analysis",
                    children=[                        
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    children=[
                                                        dbc.Col(dcc.Dropdown(id='multi-features', options=[{'label': feature, 'value': feature} for feature in data.select_dtypes(['int64', 'float64'])], value=['eta','subtotal','total_items','avg_price','delivery_distance'], style={'borderColor': '#FFD500'}, multi=True), width=12),
                                                    ],
                                                    justify='center'
                                                ),
                                                dcc.Graph(id='multi-feat-heat', config={'displayModeBar': False})
                                            ]
                                        ), className='multi-feat-heat'
                                    ), width=6
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            dcc.Graph(id='violin-plot', config={'displayModeBar': False})
                                            ),
                                        className='violin-card'
                                    ),
                                    width=6
                                )
                            ],
                            justify='center'
                        ),
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody([
                                            dcc.Dropdown(id='numeric-columns-dropdown_qq', options=[{'label': col, 'value': col} for col in data.select_dtypes(['int64', 'float64']).columns], value='eta',style={'borderColor': '#FFD500'}),
                                            dcc.Graph(id='qq-plot', config={'displayModeBar': False})
                                            ]),
                                        className='qq-card'
                                    ),
                                    width=4
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody([
                                            dcc.Dropdown(id='numeric-columns-dropdown_kde', options=[{'label': col, 'value': col} for col in data.select_dtypes(['int64', 'float64']).columns], value='delivery_distance',style={'borderColor': '#FFD500'}),
                                            dcc.Graph(id='kernel-density-plot', config={'displayModeBar': False})
                                            ]),
                                        className='kernel-density-card'
                                    ),
                                    width=4
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody([
                                            dcc.Dropdown(id='numeric-columns-dropdown_dist', options=[{'label': col, 'value': col} for col in data.select_dtypes(['int64', 'float64']).columns], value='subtotal',style={'borderColor': '#FFD500'}),
                                            dcc.Graph(id='dist-plot', config={'displayModeBar': False})
                                            ]),
                                        className='dist-plot-card'
                                    ), 
                                    width=4                                
                                ),
                            ],justify='center'
                        ),
                        dbc.Row(
                            children=[
                                 dbc.Card(
                                        dbc.CardBody(
                                            children=[
                                            dcc.Graph(id = 'box-plot', config={'displayModeBar': False})
                                            ]
                                        ),className= 'box-plot-card'
                                    )
                            ]
                        )
                    ], 
                    tab_style={ 'width': '50%', 'text-align': 'center','font-family': 'Avenir','font-size': 20}, 
                    label_style={"color": "#FFD500"},active_tab_style={'border-top': '3px solid #FFD500'}
                )
            ],
        ),
    ]
)

########################## Top Cards ##########################

def total_orders_count(data: pd.DataFrame) -> int:
    return len(data)

def average_order_size(data: pd.DataFrame) -> float:
    return round(data['total_items'].mean(),0)

def top_cousins(data: pd.DataFrame, n: int = 5) -> list:
    top_cousins_series = data['store_primary_category'].value_counts().head(n)
    return top_cousins_series.items()

def average_order_value(data: pd.DataFrame) -> float:
    return round(data['subtotal'].mean(),2)


def top_categories_list_items(data: pd.DataFrame, n: int = 5) -> list:
    top_categories_series = data['store_primary_category'].value_counts().head(n)
    list_items = []
    for category, count in top_categories_series.items():
        list_items.append(
            html.Li(
                [
                    html.Span(category.capitalize(), className='category-name'),
                    html.Span(count, className='order-count'),
                ],
                className='top-categories-list-item'
            )
        )
    return list_items



@app.callback(
    [
        Output('total-orders-card', 'children'),
        Output('average-order-size-card', 'children'),
        Output('top-category-bubble-chart', 'figure'),
        Output('average-order-value-card', 'children')
    ],
    [Input('dummy-input', 'children')]
)

def update_cards(dummy_input):
    total_orders = total_orders_count(data)
    avg_order_size = average_order_size(data)
    top_5_categories = top_cousins(data)
    avg_order_value = average_order_value(data)

    # Create bubble chart
    top_5_categories_df = pd.DataFrame(top_5_categories, columns=['category', 'count'])
    colors = ['#FFD500', '#EEAF61', '#FB9062', '#EE5D6C', '#CE4993']

    bubble_chart = px.scatter(top_5_categories_df,
                              x='category',
                              y='count',
                              size='count',
                              text=top_5_categories_df['category'], 
                              color=colors,
                              height=150,
                              width=400,
                              )
    bubble_chart.update_traces(
        textposition='middle center',
        textfont=dict(size=10, color='#FFD500'),
        hovertemplate="<b>%{text}</b><br>%{y}<extra></extra> Orders",
        showlegend=False,
        hoverinfo='text',
        mode='markers+text',
        marker=dict(sizemode='diameter', sizeref=0.012 * max(top_5_categories_df['count']), line_width=1, line_color='white')
    )

    bubble_chart.update_layout(
        plot_bgcolor='#303030',
        paper_bgcolor='#303030',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return (
        html.H1(f"{total_orders:,}", className='font-weight-bold'),
        html.H1(f"{avg_order_size:.0f}", className='font-weight-bold'),
        bubble_chart,
        html.H1(f"${avg_order_value:.2f}", className='font-weight-bold'),
    )



########################## Trend by City ##########################

def create_line_graph(df):
    df_trend = df.copy()

    df_trend['order_date'] = df_trend['created_at'].dt.date
    df_trend['num_orders'] = 1
    df_trend = df_trend.groupby(['order_date', 'market_id']).sum('num_orders').reset_index()

    fig = px.line(data_frame=df_trend, x="order_date", y="num_orders", color="market_id")
    fig.update_xaxes(title_text='Order Date',color="#FFD500")
    fig.update_yaxes(title_text='Number of Orders',color="#FFD500")
    fig.update_layout(
        legend_title='Market (City)',
        title={
            'text': 'Trend of Total Orders Over Time by City',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font_family': 'Avenir',
            'font_size': 30,
        },
        title_font_color= '#FFD500',
        font_family='Avenir',
        font_size=12,
        plot_bgcolor='#303030',
        paper_bgcolor='#303030'
    )
    return fig

@app.callback(
    Output('line-graph-city', 'figure'),
    [Input('dummy-input', 'children')]
)
def update_line_graph(dummy_input):
    return create_line_graph(data)

########################## All Feature Histogram ##########################

def all_feat_hist(data, column, color):
    fig = px.histogram(data_frame=data, x=column, nbins=500, title=f"Peering into {column}: Unraveling the Histogram", barmode='overlay', color_discrete_sequence=[color], height=410)
    fig.update_layout(title={'text': f'Histogram for {column}',
                             'x': 0.5,
                             'xanchor': 'center',
                             'y': 0.95,
                             'yanchor': 'top'},
                    plot_bgcolor='#303030',
                    paper_bgcolor='#303030',
                    title_font_color = '#FFD400',
                     font_family='Avenir',
                     font_size=12)
    fig.update_yaxes(title_text='Frequency',color="#FFD500")
    fig.update_xaxes(title_text=column,color="#FFD500")
    return fig

@app.callback(
    Output('all-feature-histogram', 'figure'),
    [Input('numeric-columns-dropdown', 'value')]
)
def update_histogram(selected_column):
    colors = px.colors.qualitative.Pastel
    random_color = colors[np.random.randint(0, len(colors))]
    return all_feat_hist(data, selected_column, random_color)   

########################## Pie Chart ##########################

def create_pie_chart(data):
    fig = px.sunburst(data, 
                      path=['time_of_day', 'delivery_nature'], 
                      color='delivery_nature', 
                      color_discrete_sequence=px.colors.qualitative.Plotly_r,
                      height=446)

    fig.update_layout(
        title={
            'text': 'Sunrise to Sunset: The Cycle of Delivery Natures and Time',
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'
        },
        font_family='Avenir',
        font_size=12,
        plot_bgcolor='#303030',
        paper_bgcolor='#303030',
        title_font_color='#FFD400'
    )

    return fig

@app.callback(Output('pie-chart', 'figure'),
              [Input('dummy-input', 'children')])

def update_pie_chart(dummy_input):
    return create_pie_chart(data=data)

########################## Count by Day and Time ##########################

def count_day_time(data):
    df_day_time_trend = data.copy()

    df_day_time_trend['num_orders'] = 1
    df_day_time_trend = df_day_time_trend.groupby(['day_of_week', 'time_of_day']).sum('num_orders').reset_index()


    fig = px.bar(df_day_time_trend, 
                 x="day_of_week", 
                 y="num_orders", 
                 color="time_of_day", 
                 hover_data=['day_of_week', 'time_of_day'], 
                 color_discrete_sequence= px.colors.qualitative.Plotly_r)
    
    fig.update_xaxes( title_text = 'Day of the Week', color = "#FFD500",categoryorder = 'array', categoryarray = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday', 'Saturday', 'Sunday'])
    fig.update_yaxes(title_text = 'Number of Orders', color = "#FFD500")
    fig.update_layout(legend_title = 'Time of the Day', 
                    title= {'text': 'A Tale of Many Cities: Order Trends Through Time',
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top','font_family': 'Avenir','font_size': 20},                    
                    font_family= 'Avenir',
                    font_size = 12,
                    plot_bgcolor='#303030',
                    paper_bgcolor='#303030',
                    title_font_color = '#FFD500',
                    xaxis=dict(showgrid=False, zeroline=False, visible=True, tickangle=-45),
)
    return fig

@app.callback(
    Output('count-day-time', 'figure'),
    [Input('dummy-input', 'children')]
)

def update_count_day_time(dummy_input):
    return count_day_time(data)

########################## Scatter Plot for all features ##########################

def all_scatter_plot(data, first_feature, second_feature):
    fig = px.scatter(
        data_frame=data,
        y=first_feature,
        x=second_feature,
        height=420,
        trendline='ols',
        trendline_color_override='white'
    )
    fig.update_layout(
        title= {'text': f'Scatter Plot for {first_feature} and {second_feature}',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top','font_family': 'Avenir','font_size': 20},         
        font_family='Avenir',
        font_size=12,
        plot_bgcolor='#303030',
        paper_bgcolor='#303030',
        title_font_color='#FFD500'
    )
    fig.update_yaxes(title_text=first_feature, color="#FFD500")
    fig.update_xaxes(title_text=second_feature, color="#FFD500")
    return fig



column_options = [{'label': col, 'value': col} for col in data.select_dtypes(['int64','float64']).columns]

@app.callback(
    Output('first-feature', 'options'),
    [Input('dummy-input', 'children')]
)
def update_first_feature_options(dummy_input):
    return column_options

@app.callback(
    Output('second-feature', 'options'),
    [Input('dummy-input', 'children')]
)
def update_second_feature_options(dummy_input):
    return column_options

@app.callback(
    Output('all-scatter-plot', 'figure'),
    [Input('first-feature', 'value'),
     Input('second-feature', 'value')]
)
def update_all_scatter_plot(first_feature, second_feature):
    if first_feature is None or second_feature is None:
        return go.Figure()
    return all_scatter_plot(data, first_feature=first_feature, second_feature=second_feature)

########################## Bubble Chart for All Categories ##########################

def create_bubble_chart(data, show_cat):
    store_categories = data['store_primary_category'].value_counts().head(show_cat).reset_index()
    store_categories.columns = ['store_primary_category', 'count']
    
    colors = px.colors.qualitative.Plotly
    fig = px.scatter(store_categories,
                     x='store_primary_category',
                     y='count',
                     size='count',
                     text='store_primary_category',
                     color='store_primary_category',
                     color_discrete_sequence=colors,
                     height=430)    

    fig.update_traces(
        textposition='top center',
        textfont=dict(size=12, color='#303030'),
        hovertemplate="<b>%{text}</b><br>Orders: %{y}<extra></extra>",
        showlegend=False,
        hoverinfo='text',
        mode='markers+text',
        marker=dict(sizemode='diameter', sizeref=0.01 * max(store_categories['count']), line_width=1, line_color='white')
    )


    
    fig.update_layout(
        title= {'text': f'Top {show_cat} Store Categories',
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top','font_family': 'Avenir','font_size': 20},         
        font_family='Avenir',
        font_size=12,
        plot_bgcolor='#303030',
        paper_bgcolor='#303030',
        title_font_color='#FFD400',
        xaxis=dict(showgrid=False, zeroline=False, visible=True, tickangle=-45),
        yaxis=dict(showgrid=False, zeroline=False, visible=True),
    )
    fig.update_yaxes(title_text='Orders', color="#FFD500")
    fig.update_xaxes(title_text='Cousins', color="#FFD500")

    return fig

@app.callback(
    Output('cat-bubble-plot','figure'),
    [Input('select-top-cat','value')]
)
def update_bubble_chart(show_cat):
    return create_bubble_chart(data=data,show_cat=show_cat)

########################## Heat Map ##########################

def create_heat_map(data, features):
    df_heatmap = data[features]
    corr_matrix = df_heatmap.corr().round(2)

    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='peach')

    fig.update_layout(title={'text': 'Correlation Matrix', 'font_family': 'Avenir', 'font_size': 20, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      font_family='Avenir',
                      font_size=12,
                      plot_bgcolor='#303030',
                      paper_bgcolor='#303030',
                      title_font_color='#FFD500'
                      )

    fig.update_yaxes( color="#FFD500")
    fig.update_xaxes(color="#FFD500",tickangle=-45)

    return fig

@app.callback(
    Output('multi-feat-heat', 'figure'),
    [Input('multi-features', 'value')]
)

def update_heat_map(features):
    return create_heat_map(data=data, features=features)

########################## Violin  Map ##########################

def create_violin_plot(data, column, x):
    fig = px.violin(data, x=x, y=column, box=True, points='all',  title=f"Violin plot of Total Busy Riders vs Delivery Duration", height=490,width=830, color_discrete_sequence = ['coral'])

    fig.update_layout(
        title={'font_family': 'Avenir', 'font_size': 20, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        font_family='Avenir',
        font_size=12,
        plot_bgcolor='#303030',
        paper_bgcolor='#303030',
        title_font_color='#FFD500',
        yaxis=dict(showgrid=False, zeroline=False, visible=True),
    )
    fig.update_yaxes(title_text='Total Busy Riders', color="#FFD500")
    fig.update_xaxes(
        title_text='Delivery Duration', 
        color="#FFD500", 
        categoryorder='array', 
        categoryarray=['<15 mins', '15 - 30 mins', '30 - 45 mins', '45 - 60 mins', '> 60 mins']
    )

    return fig


@app.callback(
    Output('violin-plot', 'figure'),
    [Input('dummy-input', 'children')]
)
def update_violin_plot(_):
    return create_violin_plot(data=data, column='total_busy_dashers', x='delivery_duration')

########################## QQ Plot ##########################

def create_qq_plot(data, col,column):

    theoretical_quantiles = sorted([stats.norm.ppf((i + 1) / (len(data[column]) + 1)) for i in range(len(data[column]))])
    
    # Sort the data in ascending order
    sorted_data = sorted(data[column])

    # Create QQ plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name='QQ Plot', fillcolor=col))
    fig.update_layout(
        title={'text':'Quantile-Quantile Plot',                            
               'x': 0.5,
               'xanchor': 'center',
               'yanchor': 'top',
               'font_family': 'Avenir',
               'font_size': 20},
        font_family='Avenir',
        font_size=12,
        plot_bgcolor='#303030',
        paper_bgcolor='#303030',
        title_font_color='#FFD500'
    )
    fig.update_traces(marker= dict(color= 'cyan'))
    fig.add_shape(type='line',
                  x0=min(theoretical_quantiles), x1=max(theoretical_quantiles),
                  y0=min(sorted_data), y1=max(sorted_data),
                  yref='y', xref='x',
                  line=dict(color='#FFFFFF', width=2))
    
    fig.update_yaxes(title_text='Theoretical Quantiles', color="#FFD500")
    fig.update_xaxes(title_text='Sample Quantiles',color="#FFD500")
    return fig



@app.callback(
    Output('qq-plot', 'figure'),
    Input('numeric-columns-dropdown_qq', 'value')
)
def update_qq_plot(column):
    colors = px.colors.qualitative.Pastel
    random_color = colors[np.random.randint(0, len(colors))]
    return create_qq_plot(data=data, col = random_color,column=column)

########################## KDE Plot ##########################

def create_kernel_density_plot(data,col, column):
    # Create kernel density plot
    fig = px.density_contour(data, x=column,color_discrete_sequence=[col])
    fig.update_layout(
        title={'text': "Kernel Density Plot",
               'x': 0.5,
               'xanchor': 'center',
               'yanchor': 'top',
               'font_family': 'Avenir',
               'font_size': 20},
        font_family='Avenir',
        font_size=12,
        plot_bgcolor='#303030',
        paper_bgcolor='#303030',
        title_font_color='#FFD500'
    )
    fig.update_yaxes(title_text='Density', color="#FFD500")
    fig.update_xaxes(title_text=column,color="#FFD500")
    return fig

@app.callback(
    Output('kernel-density-plot', 'figure'),
    Input('numeric-columns-dropdown_kde', 'value')
)
def update_kernel_density_plot(column):
    colors = px.colors.qualitative.Pastel
    random_color = colors[np.random.randint(0, len(colors))]    
    return create_kernel_density_plot(data=data,col=random_color, column=column)

########################## Dist Plot ##########################

def create_dist_plot(data, column,col):
    fig = ff.create_distplot([data[column]], [column], bin_size=0.5, show_curve=True, colors=[col])
    fig.update_layout(
        title={'text': f'Distplot of {column}',
               'x': 0.5,
               'xanchor': 'center',
               'yanchor': 'top',
               'font_family': 'Avenir',
               'font_size': 20},
        font_family='Avenir',
        font_size=12,
        plot_bgcolor='#303030',
        paper_bgcolor='#303030',
        title_font_color='#FFD500'
    )
    fig.update_yaxes(title_text='Density', color="#FFD500")
    fig.update_xaxes(title_text=column, color="#FFD500")
    
    return fig

@app.callback(
    Output('dist-plot', 'figure'),
    [Input('numeric-columns-dropdown_dist', 'value')]
)
def update_dist_plot(feature):
    colors = px.colors.qualitative.Pastel
    random_color = colors[np.random.randint(0, len(colors))]
    return create_dist_plot(data=data, column=feature,col=random_color)

########################## Box Plot ##########################

def create_multi_box(data):
    fig = px.box(data_frame= data,
                 x= 'day_of_week',
                 y='subtotal',
                 color='delivery_nature')
    fig.update_layout(
        legend_title= "Delivery Type",
        title = {
        'text': 'Multivariate Box Plot for Day of the Week By Delivery Type',
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font_family': 'Avenir',
        'font_size': 20},
        font_family = 'Avenir',
        font_size = 12,
        plot_bgcolor = '#303030',
        paper_bgcolor = '#303030',
        title_font_color = '#FFD500'
    )
    fig.update_xaxes(title_text = 'Day of the Week',color = "#FFD500")
    fig.update_yaxes(title_text = 'Order Value',color = "#FFD500")
    return fig

@app.callback(
    Output('box-plot', 'figure'),
    [Input('dummy-input', 'children')]
)
def update_box_plot(dummy_input):
    return create_multi_box(data=data)


if __name__ == '__main__':
     app.run_server(debug=True,host = '0.0.0.0',  port = 8002)
