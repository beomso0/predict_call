# import libraries
from asyncore import close_all
from base64 import encode
from cProfile import label
from concurrent.futures import process
from distutils.command.upload import upload
import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import ast
import math
import datetime
# from pycaret.regression import *
import s3fs
import os
import pickle
# import catboost
import lightgbm as lgb
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
import preprocess

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
if 'df_input' not in st.session_state:
     st.session_state.df_input = pd.DataFrame(columns=[
          'year_input',
          'month_input',
          'weekday_input',
          'holiday_input',
          'weather_input',
          'time_input',
          'duration_input',
          'showhost_input',
          'expression_input',
          'live_input',
          'midcat_input',
          'brand_input',
          'price_input',
     ])

# 전처리 된 테이블 생성
if 'df_preprocessed' not in st.session_state:
     st.session_state.df_preprocessed = None
if 'is_processed' not in st.session_state:
     st.session_state.is_processed = 0

# title
st.title('PGM별 콜 예측')

@st.cache(ttl=6000)
def load_model(model_name, encoder_name):
     return joblib.load(model_name), joblib.load(encoder_name)

@st.cache(persist=True)
def load_ref(ref_name):
     return joblib.load(ref_name)

@st.cache(ttl=6000)
def apply_backup(backup):
     st.session_state.df_input = backup

model_load_state = st.text('Loading model and encoder...')
# model = read_file("hhxgh/model_compressed.pkl")
# model, encoder = load_model('jh_caret/0214_final_reg_saved.pkl','0214_encoder.pkl')
model_load_state.text('Model and encoder loaded!')

ref_load_state = st.text('Loading Ref...')
if 'ref' not in st.session_state:
     st.session_state.ref = load_ref('ref.pkl')
ref_load_state.text('Ref loaded!')

with st.form("백업파일 업로드", clear_on_submit=True):
     file = st.file_uploader("백업 파일을 드래그하여 업로드하세요. 업로드 시 현재 데이터는 사라지니 주의해주세요.")
     submitted = st.form_submit_button("업로드 및 적용")

     if submitted and file is not None:
          st.session_state.df_input = joblib.load(file)


# uploaded_file = st.file_uploader("Drag and drop a file", on_change=apply_backup)
# if uploaded_file is not None:
#      # Can be used wherever a "file-like" object is accepted:
#      # dataframe = pd.read_csv(uploaded_file)
#      st.session_state.df_input = joblib.load(uploaded_file)
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

@st.cache(ttl=600)
def del_row(del_idx):
     return st.session_state.df_input.drop(labels=range(del_idx[0],del_idx[1]+1),axis=0).reset_index(drop=True)


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
             
          year_input = st.number_input(
               '연도', 2022,2027
          )
          month_input = st.number_input(
               '월', 1,12
          )
          weekday_input = st.selectbox(
               '요일', st.session_state.ref['weekday_ref']
          )
          holiday_input = st.checkbox(
               '주말 혹은 공휴일 여부',
          )    
          weather_input = st.selectbox(
               '예상 날씨', st.session_state.ref['weather_ref']
          )
          time_input = st.number_input(
               '방송 시작 시각(0~24)', 0,24
          )
          duration_input = st.number_input(
               '방송 길이(분)', 0,1440
          )
          showhost_input = st.multiselect(
               '출연 쇼호스트(전체)', st.session_state.ref['showhost_ref']
          )         
          expression_input = st.multiselect(
               '사용 예정인 한정표현(전체)', st.session_state.ref['expression_ref']
          ) 
          live_input = st.checkbox(
               'LIVE 방송 여부', True
          )
          # prd_num_input = st.number_input(
          #      '전체 판매 상품 개수', 0
          # )
          st.write('**----------------------------------------------------**')
          st.write('**각 상품의 중분류-브랜드-상품가격을 순서를 맞추어 입력해주세요**')
          # st.write('**중복되는 브랜드-중분류 조합은 입력하지 않아도 됩니다.**')
          midcat_input = st.multiselect(
               '판매 상품 중분류(전체)', st.session_state.ref['midcat_ref']
          ) 
          brand_input = st.multiselect(
               '판매 상품 브랜드(전체)', st.session_state.ref['brand_ref']
          )
          price_input = st.multiselect(
               '상품가격(전체)(천원)', st.session_state.ref['price_ref']
          )
          st.write('**----------------------------------------------------**') 

          # submit
          submitted = st.form_submit_button('Submit')
          # update dataframe

          if submitted:
               print(showhost_input)
               st.session_state.df_input = st.session_state.df_input.append({
                                                                           'showhost_input':showhost_input,
                                                                           'weather_input':weather_input,
                                                                           'weekday_input':weekday_input,
                                                                           'brand_input':brand_input,
                                                                           'midcat_input':midcat_input,
                                                                           'expression_input':expression_input,
                                                                           'year_input':year_input,
                                                                           'month_input':month_input,
                                                                           'time_input':time_input,
                                                                           'duration_input':duration_input,
                                                                           'live_input':live_input,
                                                                           'holiday_input':holiday_input,
                                                                           'price_input':price_input
                                                                 },ignore_index=True)

col1, col2= st.columns([1.5,9])
with col1:    
     save = st.download_button(
          label='임시저장',
          data= pickle.dumps(st.session_state.df_input),
          file_name=f'{str(datetime.datetime.now())[:-7]}_backup.pkl',
     )             
     predict = st.button('예측')
     if len(st.session_state.df_input)!=0:  
          delete_row = st.button('행 삭제하기')
with col2:
     # 삭제 target row index slider
     if len(st.session_state.df_input)!=0:
          del_idx = st.slider(
               '삭제할 행을 선택해주세요', 
               0,
               100,
               (0,1),
               key='del_slider'
          )
          if delete_row:
               try:
                    st.session_state.df_input = st.session_state.df_input.drop(labels=list(range(del_idx[0],del_idx[1]+1)),axis=0).reset_index(drop=True)
               except:
                   st.error('삭제 행의 범위를 다시 확인해주세요') 

if predict:
     try:
          st.session_state.df_preprocessed = preprocess.process_input(st.session_state.df_input)
          st.session_state.is_processed = 1
     except Exception as e:
          st.error(e)

# show dataframe
st.subheader('입력된 데이터')
st.dataframe(st.session_state.df_input.style.format(precision=0))

if st.session_state.is_processed == 1:
     st.subheader('전처리된 데이터')
     st.dataframe(st.session_state.df_preprocessed)
# if predict:
#      st.dataframe(models())
