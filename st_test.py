# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from category_encoders.cat_boost import CatBoostEncoder
import joblib 
import ast
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import math
import datetime

# declare global variables
st.set_page_config(layout='wide')
file_uploaded = 0
preprocess_done = 0
predict_done = 0 
if 'df_to_predict' not in st.session_state:
     st.session_state.df_to_predict = pd.DataFrame(columns=[
          'pgm_input',
          'showhost_input',
          'weather_input',
          'weekday_input',
          'brand_input',
          'expression_input',
          'date_input',
          'time_input',
          'duration_input',
          'prd_num_input',
          'live_input',
          'holiday_input'
     ])

st.title('모델 배포 테스트')

@st.cache
def load_model(model_name):
     return joblib.load(model_name)

@st.cache
def load_ref(ref_name):
     return joblib.load(ref_name)


model_load_state = st.text('Loading model...')
model = load_model('test_lgb.pkl')
model_load_state.text('Model loaded!')

ref_load_state = st.text('Loading Ref...')
ref = load_ref('ref.pkl')
ref_load_state.text('Ref loaded!')

uploaded_file = st.file_uploader("Drag and drop a file")
if uploaded_file is not None:
     # Can be used wherever a "file-like" object is accepted:
     dataframe = pd.read_csv(uploaded_file)
     st.write(dataframe)
     file_uploaded = 1


@st.cache
def preprocess_df(raw_df):
     pass

@st.cache
def make_pred(model,target_df):
     target_inv_trans = lambda call_: np.expm1(call_ ** (10/16))
     target_df['예측값'] = target_inv_trans(model.predict(target_df))
     return target_df

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

if (file_uploaded == 1) & (preprocess_done == 1):
     output = make_pred(model,dataframe)
     predict_done = 1

if predict_done == 1:   
     st.header('예측완료!')
     st.write(output)

     final_csv = convert_df(output)

     st.download_button(
          label="Download Prediction Table",
          data=final_csv,
          file_name='prediction.csv',
          mime='text/csv',
     )

def update_df(
          pgm_input,
          showhost_input,
          weather_input,
          weekday_input,
          brand_input,
          expression_input,
          date_input,
          time_input,
          duration_input,
          prd_num_input,
          live_input,
          holiday_input
          ):
     st.session_state.df_to_predict.append({
               'pgm_input':pgm_input,
               'showhost_input':showhost_input,
               'weather_input':weather_input,
               'weekday_input':weekday_input,
               'brand_input':brand_input,
               'expression_input':expression_input,
               'date_input':date_input,
               'time_input':time_input,
               'duration_input':duration_input,
               'prd_num_input':prd_num_input,
               'live_input':live_input,
               'holiday_input':holiday_input
     },ignore_index=True)


col1, col2 = st.columns([1,3])

with col1:
#test
     with st.form(key='columns_in_form'):
          pgm_input = st.selectbox(
               'PGM명을 선택하세요', ref['pgm_ref']
          )
          showhost_input = st.multiselect(
               '출연 쇼호스트를 모두 선택하세요', ref['showhost_ref']
          )
          weather_input = st.selectbox(
               '예상 날씨를 선택하세요', ref['weather_ref']
          )
          weekday_input = st.selectbox(
               '요일을 선택하세요', ref['weekday_ref']
          )
          brand_input = st.multiselect(
               '판매 브랜드를 모두 선택하세요', ref['brand_ref']
          )
          expression_input = st.multiselect(
               '사용 예정인 한정표현을 모두 선택하세요', ref['expression_ref']
          )    
          date_input = st.date_input(
               '방송시작일자', datetime.date.today()+datetime.timedelta(7)
          )
          time_input = st.number_input(
               '방송시작시간(0~24)', 0,24
          )
          duration_input = st.number_input(
               '방송 길이(분)', 0,1440
          )
          prd_num_input = st.number_input(
               '판매 상품 개수', 0
          )
          live_input = st.checkbox(
               'LIVE 방송인가요?', True
          )
          holiday_input = st.checkbox(
               '주말 혹은 공휴일인가요?',
          )    

          # 제출
          submitted = st.form_submit_button('Submit',on_click=update_df,args=[
               pgm_input,
               showhost_input,
               weather_input,
               weekday_input,
               brand_input,
               expression_input,
               date_input,
               time_input,
               duration_input,
               prd_num_input,
               live_input,
               holiday_input
          ])

with col2:
     st.write(st.session_state.df_to_predict)

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


