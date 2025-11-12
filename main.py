# streamlit app for dashboard and prediction

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# Set page layout to wide for maximum screen width
st.set_page_config(layout="wide")

API_URL = "http://localhost:8000/predict/{n}"

from src.utils import load_data_from_s3

st.title("USD VND Prediction")

# Load data with caching
df = st.cache_data(ttl=3600)(load_data_from_s3)()

# Define predefined time range - only show last 6 months of data
end_date = df.index.max()
start_date = end_date - pd.Timedelta(days=180)  # Last 6 months
chart_data = pd.DataFrame(df['Lần cuối'][(df.index >= start_date) & (df.index <= end_date)])

# Initialize session state for predictions
if 'predictions_data' not in st.session_state:
    st.session_state.predictions_data = None
if 'last_n' not in st.session_state:
    st.session_state.last_n = None

# Display time range info
st.info(f"📅 Showing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (Last 6 months)")

# prediction input
n = st.number_input("Enter the number of days to predict", value=10, min_value=1, max_value=100)

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Predict"):
        with st.spinner('Fetching predictions...'):
            response = requests.get(API_URL.format(n=n), timeout=10)
            if response.status_code == 429:
                st.error("Too many requests. Please try again later.")
                # skip the rest of the code
                pass
            else:
                predictions = response.json() # -> list[dict]
                predictions_data = pd.DataFrame(predictions)
                predictions_data.set_index('Ngày', inplace=True)
                predictions_data.index = pd.to_datetime(predictions_data.index)
                st.session_state.predictions_data = predictions_data
                st.session_state.last_n = n
                st.rerun()
    
    # Clear predictions button
    if st.session_state.predictions_data is not None:
        if st.button("Clear Predictions"):
            st.session_state.predictions_data = None
            st.session_state.last_n = None
            st.rerun()

# Use session state predictions
predictions_data = st.session_state.predictions_data

# Create combined chart with actual and prediction data
fig = go.Figure()

# Add actual data trace
fig.add_trace(go.Scatter(
    x=chart_data.index,
    y=chart_data['Lần cuối'],
    mode='lines',
    name='Actual',
    line=dict(color='blue', width=2)
))

# Add prediction data traces if available
if predictions_data is not None:
    for col in predictions_data.columns:
        fig.add_trace(go.Scatter(
            x=predictions_data.index,
            y=predictions_data[col],
            mode='lines',
            name=f'Prediction',
            line=dict(color='red', width=2)
        ))

# Calculate y-axis range to fit both actual and prediction data
y_min = chart_data['Lần cuối'].min()
y_max = chart_data['Lần cuối'].max()

if predictions_data is not None:
    pred_min = predictions_data.min().min()
    pred_max = predictions_data.max().max()
    y_min = min(y_min, pred_min)
    y_max = max(y_max, pred_max)

# Update layout with combined data range
fig.update_layout(
    title='USD VND: Actual vs Predictions',
    xaxis_title='Ngày',
    yaxis_title='Value',
    yaxis=dict(range=[y_min, y_max]),
    height=500,
    hovermode='x unified',
    autosize=True,
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig, width='stretch', config={'displayModeBar': True})