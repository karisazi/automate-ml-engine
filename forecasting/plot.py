import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Create random data with numpy
import numpy as np

# df = pd.read_csv('dataset.csv')

# target_col = 'pollution'

def plot_actual_vs_forecast(df, date_df, target_col):
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date_df, y=df['ground_truth'],
                        mode='lines',
                        name='actual'))
    fig.add_trace(go.Scatter(x=date_df, y=df['prediction'],
                        mode='lines',
                        name='prediction'))
    
    # # use it to custom x values
    
    # fig.update_xaxes(
    #         rangeslider_visible=True,
    #         rangeselector=dict(
    #             buttons=list(
    #                 [
    #                     dict(count=7, label="1w", step="day", stepmode="backward"),
    #                     dict(count=1, label="1m", step="month", stepmode="backward"),
    #                     dict(count=3, label="3m", step="month", stepmode="backward"),
    #                     dict(count=6, label="6m", step="month", stepmode="backward"),
    #                     dict(count=1, label="YTD", step="year", stepmode="todate"),
    #                     dict(count=1, label="1y", step="year", stepmode="backward"),
    #                     dict(step="all"),
    #                 ]
    #             )
    #         ),
    #     )
    # Edit the layout
    fig.update_layout(title='Forecast vs Truth',
                    xaxis_title='Date',
                    yaxis_title=target_col)

    st.plotly_chart(fig)