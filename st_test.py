# import libraries
from asyncore import close_all
from base64 import encode
from cProfile import label
from distutils.command.upload import upload
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from category_encoders.cat_boost import CatBoostEncoder
import joblib 
import ast
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import datetime
from pycaret.regression import *
import s3fs
import os
import pickle

# create S3 file system connection object
fs = s3fs.S3FileSystem(anon=False)

# Retrieve file contents
@st.cache(ttl=6000)
def read_file(filename):
    with fs.open(filename) as f:
        return joblib.load(f)


# set overall layout
st.set_page_config(layout='wide')

# declare session states
# if 'file_uploaded' not in st.session_state:
#      st.session_state.file_uploaded = 0
if 'preprocess_done' not in st.session_state:
     st.session_state.preprocess_done = 0
if 'predict_done' not in st.session_state:
     st.session_state.predict_done = 0

# dataframe 생성
if 'df_to_predict' not in st.session_state:
     st.session_state.df_to_predict = pd.DataFrame(columns=[
          'showhost_input',
          'weather_input',
          'weekday_input',
          'brand_input',
          'midcat_input',
          'price_input',
          'expression_input',
          'date_input',
          'time_input',
          'duration_input',
          'prd_num_input',
          'live_input',
          'holiday_input'
     ])

st.title('모델 배포 테스트')

@st.cache(ttl=6000)
def load_model(model_name, encoder_name):
     return joblib.load(model_name), joblib.load(encoder_name)

@st.cache(ttl=6000)
def load_ref(ref_name):
     return joblib.load(ref_name)

@st.cache
def apply_backup(backup):
     st.session_state.df_to_predict = backup

model_load_state = st.text('Loading model and encoder...')
# model = read_file("hhxgh/model_compressed.pkl")
# model, encoder = load_model('jh_caret/0214_final_reg_saved.pkl','0214_encoder.pkl')
model_load_state.text('Model and encoder loaded!')

ref_load_state = st.text('Loading Ref...')
ref = load_ref('ref.pkl')
ref_load_state.text('Ref loaded!')

with st.form("백업파일 업로드", clear_on_submit=True):
     file = st.file_uploader("백업 파일을 드래그하여 업로드하세요. 업로드 시 현재 데이터는 사라지니 주의해주세요.")
     submitted = st.form_submit_button("업로드 및 적용")

     if submitted and file is not None:
          st.session_state.df_to_predict = joblib.load(file)


# uploaded_file = st.file_uploader("Drag and drop a file", on_change=apply_backup)
# if uploaded_file is not None:
#      # Can be used wherever a "file-like" object is accepted:
#      # dataframe = pd.read_csv(uploaded_file)
#      st.session_state.df_to_predict = joblib.load(uploaded_file)
#      uploaded_file = None

@st.cache(ttl=6000)
def preprocess_df(raw_df):
     pass

@st.cache(ttl=6000)
def make_pred(model,target_df):
     target_inv_trans = lambda call_: np.expm1(call_ ** (10/16))
     target_df['예측값'] = target_inv_trans(model.predict(target_df))
     return target_df

@st.cache(ttl=600)
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

if st.session_state.preprocess_done == 1:
     output = make_pred(model,dataframe)
     st.session_state.predict_done = 1

if st.session_state.predict_done == 1:   
     st.header('예측완료!')
     st.write(output)

     final_csv = convert_df(output)

     st.download_button(
          label="Download Prediction Table",
          data=final_csv,
          file_name='prediction.csv',
          mime='text/csv',
     )

# sidebar form
with st.sidebar.form(key='columns_in_form'):          

          showhost_input = st.multiselect(
               '출연 쇼호스트(전체)', ref['showhost_ref']
          )
          weather_input = st.selectbox(
               '예상 날씨', ref['weather_ref']
          )
          weekday_input = st.selectbox(
               '요일', ref['weekday_ref']
          )
          st.write('**----------------------------------------------------**')
          st.write('**브랜드-중분류는 순서를 맞춰 입력해주세요**')
          st.write('**중복되는 브랜드-중분류 조합은 입력하지 않아도 됩니다.**')
          brand_input = st.multiselect(
               '판매 상품 브랜드(전체)', ref['brand_ref']
          )
          midcat_input = st.multiselect(
               '판매 상품 중분류(전체)', ref['midcat_ref']
          )
          st.write('**----------------------------------------------------**')
          price_input = st.number_input(
               '상품가격(전체)', 0
          )
          expression_input = st.multiselect(
               '사용 예정인 한정표현(전체)', ref['expression_ref']
          )    
          date_input = st.date_input(
               '방송시작일자', datetime.date.today()+datetime.timedelta(7)
          )
          time_input = st.number_input(
               '방송 시작 시각(0~24)', 0,24
          )
          duration_input = st.number_input(
               '방송 길이(분)', 0,1440
          )
          prd_num_input = st.number_input(
               '전체 판매 상품 개수', 0
          )
          live_input = st.checkbox(
               'LIVE 방송 여부', True
          )
          holiday_input = st.checkbox(
               '주말 혹은 공휴일 여부',
          )    

          # submit
          submitted = st.form_submit_button('Submit')
          # update dataframe
          if submitted:
               print(showhost_input)
               st.session_state.df_to_predict = st.session_state.df_to_predict.append({
                                                                           'showhost_input':showhost_input,
                                                                           'weather_input':weather_input,
                                                                           'weekday_input':weekday_input,
                                                                           'brand_input':brand_input,
                                                                           'midcat_input':midcat_input,
                                                                           'expression_input':expression_input,
                                                                           'date_input':date_input,
                                                                           'time_input':time_input,
                                                                           'duration_input':duration_input,
                                                                           'prd_num_input':prd_num_input,
                                                                           'live_input':live_input,
                                                                           'holiday_input':holiday_input,
                                                                           'price_input':price_input
                                                                 },ignore_index=True)
col1, col2,col3 = st.columns([1.3,1,10])
with col1:    
     save = st.download_button(
          label='임시저장',
          data= pickle.dumps(st.session_state.df_to_predict),
          file_name=f'{str(datetime.datetime.now())[:-7]}_backup.pkl',
     )             
with col2:
     predict = st.button('예측')
# show dataframe
st.dataframe(st.session_state.df_to_predict.style.format(precision=0))

# if predict:
#      st.dataframe(models())
