# vim: expandtab tabstop=4 shiftwidth=4

import ssl

from datetime import datetime
from io import StringIO
from math import log
from urllib.request import urlopen

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

import numpy as np
import pandas as pd

def fetch_data():
    now_epoch = int(datetime.now().timestamp())
    data_url = 'https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=-1325548800&period2={today}&interval=1d&events=history&includeAdjustedClose=true'.format(today=now_epoch)
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    with urlopen(data_url, context=context) as f:
        df = pd.read_csv(StringIO(str(f.read(), encoding='utf8')))

    return df

def add_log_close(df):
    df['Log Close'] = [log(c) for c in df['Adj Close']]
    return df

def add_ordinal_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Ordinal'] = [dt.toordinal() for dt in df['Date']]
    return df

def generate_predictor(df):
    # https://data36.com/linear-regression-in-python-numpy-polyfit/
    model = np.polyfit(df['Ordinal'], df['Log Close'], 1)
    predictor = np.poly1d(model)
    return predictor

def add_predicted(df, predictor):
    df['Predicted'] = [predictor(ordinal_date) for ordinal_date in df['Ordinal']]
    return df

df = fetch_data()
df = add_log_close(df)
df = add_ordinal_date(df)
predictor = generate_predictor(df)
df = add_predicted(df, predictor)

full_fig = go.Figure()
full_fig.add_trace(go.Scatter(x=df['Date'], y=df['Log Close'], mode='lines', name='Log Close'))
full_fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted'], mode='lines', name='Predicted Close'))

recent_fig = go.Figure()
recent_fig.add_trace(go.Scatter(x=df['Date'][-365*2:], y=df['Log Close'][-365*2:], mode='lines', name='Log Close'))
recent_fig.add_trace(go.Scatter(x=df['Date'][-365*2:], y=df['Predicted'][-365*2:], mode='lines', name='Predicted Close'))

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    dcc.Graph(id='recent-timeline', figure=recent_fig),
    dcc.Graph(id='full-timeline', figure=full_fig)
])

def main():
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
