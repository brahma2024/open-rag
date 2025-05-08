# module 3: plot fintrade data
# version: 2.0
# function: plot data on a plotly webpage after fetching it from influxdb

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from influxdb_client import InfluxDBClient
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Initialize the InfluxDB client
url = os.getenv("INFLUXDB_URL")
token = os.getenv("INFLUXDB_TOKEN")
org = os.getenv("INFLUXDB_ORG")
bucket = os.getenv("INFLUXDB_BUCKET")

client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

# Function to fetch stock IDs from the stocks measurement
def fetch_available_stocks():
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: -1y)
      |> filter(fn: (r) => r._measurement == "stocks")
      |> keep(columns: ["stock_id"])
      |> distinct(column: "stock_id")
    '''
    result = query_api.query_data_frame(org=org, query=query)
    if not result.empty:
        return result['stock_id'].unique().tolist()
    return []

# Function to fetch column data dynamically from the prices measurement
def fetch_columns_from_prices():
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: -1y)
      |> filter(fn: (r) => r._measurement == "prices")
      |> limit(n: 1)
      |> keep(columns: ["_field"])
      |> distinct(column: "_field")
    '''
    result = query_api.query_data_frame(org=org, query=query)
    if not result.empty:
        columns = result['_field'].unique().tolist()
        # Exclude specific columns like 'result', 'table', etc.
        excluded_columns = {'result', 'table'}
        columns = [col for col in columns if col.lower() not in excluded_columns]
        return columns
    return []

# Fetching data dynamically
available_stocks = fetch_available_stocks()
columns = fetch_columns_from_prices()

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Stock Price Time Series Dashboard"),
    dcc.Checklist(
        id='stock-selector',
        options=[{'label': stock, 'value': stock} for stock in available_stocks],
        value=available_stocks[:2],  # Default selection
        inline=True,
        labelStyle={'margin-right': '15px'}
    ),
    dcc.Checklist(
        id='field-selector',
        options=[{'label': col.capitalize(), 'value': col} for col in columns],
        value=['close_price'] if 'close_price' in columns else columns[:1],  # Default selection
        inline=True,
        labelStyle={'margin-right': '15px'}
    ),
    html.Div(id='graph-container')
])

# Callback to update the plots based on selected stocks and fields
@app.callback(
    Output('graph-container', 'children'),
    [Input('stock-selector', 'value'), Input('field-selector', 'value')]
)
def update_graphs(selected_stocks, selected_fields):
    if not selected_stocks or not selected_fields:
        return []

    # Fetch data for all selected stocks
    stock_data_map = {stock: fetch_stock_data(stock) for stock in selected_stocks}

    graphs = []
    for field in selected_fields:
        data_traces = []
        for stock_id, stock_data in stock_data_map.items():
            if not stock_data.empty and field in stock_data:
                trace = go.Scatter(
                    x=stock_data.index,
                    y=stock_data[field],
                    mode='lines',
                    name=f"{stock_id} - {field.capitalize()}",
                    line=dict(width=2)
                )
                data_traces.append(trace)

        graph = dcc.Graph(
            id=f'graph-{field}',
            figure={
                'data': data_traces,
                'layout': {
                    'title': f"{field.capitalize()} Time Series for Selected Stocks",
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': field.capitalize()},
                    'height': 400,
                    'legend': {'orientation': 'h'}
                }
            }
        )
        graphs.append(graph)

    # Arrange graphs in tiles (3 per row)
    return [html.Div(graphs[i:i+3], style={'display': 'flex'}) for i in range(0, len(graphs), 3)]

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

# Close the client after usage (if needed)
client.close()
