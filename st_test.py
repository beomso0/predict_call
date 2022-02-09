#%%
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from category_encoders.cat_boost import CatBoostEncoder
import joblib 
import ast
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import math

st.title('모델 배포 테스트')

@st.cache
def load_model(model_name):
     return joblib.load(model_name)

model_load_state = st.text('Loading model...')
model = load_model('test_lgb.pkl')
model_load_state.text('Model loaded!')

file_uploaded = None
uploaded_file = st.file_uploader("Drag and drop a file")
if uploaded_file is not None:
     # Can be used wherever a "file-like" object is accepted:
     dataframe = pd.read_csv(uploaded_file)
     st.write(dataframe)
     file_uploaded = True

@st.cache
def preprocess_df(raw_df):
     pass

predict_done = None

@st.cache
def make_pred(model,target_df):
     pass

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

if file_uploaded:
     final_csv = convert_df(dataframe)

     st.download_button(
          label="Download Prediction Table",
          data=final_csv,
          file_name='prediction.csv',
          mime='text/csv',
     )

# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#          'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache()
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# # Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# # Load 10,000 rows of data into the dataframe.
# data = load_data(10000)
# # Notify the reader that the data was successfully loaded.
# data_load_state.text('Done! (using st.cache)')

# st.subheader('Raw data')
# st.write(data)

# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(
#     data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)

# st.subheader('Map of all pickups')
# st.map(data)

# hour_to_filter = 17
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
# st.subheader(f'Map of all pickups at {hour_to_filter}:00')
# st.map(filtered_data)

# hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h

# #test
# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)


