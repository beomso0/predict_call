#%%
# import libraries
import traceback
from asyncore import close_all
from base64 import encode
from cProfile import label
from concurrent.futures import process
from distutils.command.upload import upload
from matplotlib.pyplot import axis
from pygments import highlight
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
from catboost import CatBoostRegressor
import lightgbm as lgb
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import PowerTransformer
import preprocess


#%%
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

# input dataframe 생성
if 'df_input' not in st.session_state:
     st.session_state.df_input = pd.DataFrame(columns=[
          # 추가: 'temp_input','end_time_input', 'start_time_input', 'date_input','pgm_input
          # 삭제: 'year_input','month_input','time_input', 'weekday_input'
          'date_input',
          'holiday_input',
          'temp_input',
          'pgm_input',
          'start_time_input',
          'end_time_input',
          'duration_input',
          'showhost_input',
          'expression_input',
          'live_input',
          'midcat_input',
          'brand_input',
          'price_min_input',
          'price_max_input',
          'price_mean_input',
          'product_num_input'
     ])

# 전처리 된 테이블 생성
if 'df_preprocessed' not in st.session_state:
     st.session_state.df_preprocessed = None
if 'is_predicted' not in st.session_state:
     st.session_state.is_predicted = 0
# make predicted dataframe
if 'df_predicted' not in st.session_state:
     st.session_state.df_predicted = None

# title
st.title('PGM별 콜 예측')

@st.cache(ttl=6000)
def load_model(model_name, encoder_name, transformer_name):
     return joblib.load(model_name), joblib.load(encoder_name), joblib.load(transformer_name)

@st.cache(ttl=6000)
def make_pred(df):
     encoded_df = preprocess.process_input(df, st.session_state.score, encoder)
     return pd.Series(transformer.inverse_transform(model.predict(encoded_df).reshape(-1,1)).reshape(-1),name='prediction').apply(round)

@st.cache(persist=True)
def load_test_x(test_name):
     return joblib.load(test_name)
     
@st.cache(persist=True)
def load_test_y(test_name):
     return joblib.load(test_name)

@st.cache(persist=True)
def load_ref(ref_name):
     return joblib.load(ref_name)

@st.cache(persist=True)
def load_score(brand_name, limit_name, midcat_name):
     score_lists = [
          pd.read_pickle(midcat_name),
          pd.read_pickle(brand_name),
          pd.read_pickle(limit_name)
     ]
     return score_lists

@st.cache(ttl=6000)
def apply_backup(backup):
     st.session_state.df_input = backup

# load model, encoder, transformer
model_load_state = st.text('Loading model and encoder...')
model, encoder, transformer = load_model('model/final_model.pkl','model/cat_encoder.pkl','model/boxcox_transformer.pkl')
model_load_state.text('Model and encoder loaded!')

# load ref
ref_load_state = st.text('Loading Ref...')
if 'ref' not in st.session_state:
     st.session_state.ref = load_ref('ref.pkl')
ref_load_state.text('Ref loaded!')

# load score
score_load_state = st.text('Loading Scores...')
if 'score' not in st.session_state:
     st.session_state.score = load_score('model/brand_score.pkl','model/expression_score.pkl','model/midcat_score.pkl')
score_load_state.text('Scores loaded!')

# load test
# test_load_state = st.text('Loading Test...')
# if 'test' not in st.session_state:
#      st.session_state.test = load_test_x('model/X_test_encoded.pkl')
# if 'test_y' not in st.session_state:
#      f = load_test_y('model/y_test.pkl')
#      st.session_state.test_y = pd.DataFrame(f)
# # st.dataframe(st.session_state.test_y)
# test_load_state.text('Test loaded!')

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

@st.cache(ttl=600)
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(index=False).encode('utf-8-sig')

@st.cache(ttl=600)
def del_row(del_idx):
     return st.session_state.df_input.drop(labels=range(del_idx[0],del_idx[1]+1),axis=0).reset_index(drop=True)

if st.session_state.predict_done == 1:   
     st.header('예측완료!')
     # st.write(output)

     # final_csv = convert_df(output)

     st.download_button(
          label="Download Prediction Table",
          data=final_csv,
          file_name='prediction.csv',
          mime='text/csv',
     )

# sidebar form
with st.sidebar.form(key='columns_in_form'):       
             
          date_input = st.date_input(
               '일자', datetime.date.today() + datetime.timedelta(days=7)
          )
          holiday_input = st.checkbox(
               '공휴일 여부(주말 제외)',
          )    
          temp_input = st.number_input(
               '예상 기온', -100,100,0
          )
          pgm_input = st.number_input(
               'PGM 코드', step=1
          )
          start_time_input = st.number_input(
               '방송 시작 시각(0~24)', 0,24
          )
          end_time_input = st.number_input(
               '방송 종료 시각(0~24)', 0,24
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
          # st.write('**각 상품의 중분류-브랜드-상품가격을 순서를 맞추어 입력해주세요**')
          st.write('**중복되는 브랜드-중분류 조합은 입력하지 않아도 됩니다.**')
          midcat_input = st.multiselect(
               '판매 상품 중분류(전체)', st.session_state.ref['midcat_ref']
          ) 
          brand_input = st.multiselect(
               '판매 상품 브랜드(전체)', st.session_state.ref['brand_ref']
          )
          price_min_input = st.number_input(
               '최저 상품 가격(천원)', 0,100000
          )
          price_max_input = st.number_input(
               '최고 상품 가격(천원)', 0,100000
          )
          price_mean_input = st.number_input(
               '상품 가격 평균(천원)', 0,100000
          )
          product_num_input = st.number_input(
               '판매 상품 개수', 0,1000
          )
          st.write('**----------------------------------------------------**') 

          # submit
          submitted = st.form_submit_button('Submit')
          # update dataframe

          if submitted:
               print(showhost_input)
               st.session_state.df_input = st.session_state.df_input.append({
                                                                      'date_input':date_input,
                                                                      'holiday_input':holiday_input,
                                                                      'temp_input':temp_input,
                                                                      'pgm_input': pgm_input,
                                                                      'start_time_input':start_time_input,
                                                                      'end_time_input':end_time_input,
                                                                      'duration_input':duration_input,
                                                                      'showhost_input':showhost_input,
                                                                      'expression_input':expression_input,
                                                                      'live_input':live_input,
                                                                      'midcat_input':midcat_input,
                                                                      'brand_input':brand_input,
                                                                      'price_min_input':price_min_input,
                                                                      'price_max_input':price_max_input,
                                                                      'price_mean_input':price_mean_input,
                                                                      'product_num_input':product_num_input,
                                                                 },ignore_index=True)

col1, col2= st.columns([1.5,9])
with col1:        
     # delete button
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
          # delete row
          if delete_row:
               try:
                    st.session_state.df_input = st.session_state.df_input.drop(labels=list(range(del_idx[0],del_idx[1]+1)),axis=0).reset_index(drop=True)
               except:
                   st.error('삭제 행의 범위를 다시 확인해주세요') 

# show dataframe
col1, col2, col3, col4 = st.columns([3,1.5,1.5,10])
with col1:
     st.subheader('입력된 데이터')
with col2:
     save = st.download_button(
          label='임시저장',
          data= pickle.dumps(st.session_state.df_input),
          file_name=f'{str(datetime.datetime.now())[:-7]}_backup.pkl',
     )   
with col3:
     do_predict = st.button('예측')

# predict button
if do_predict:
     # preprocess
     try:
          st.session_state.df_predicted = pd.concat([make_pred(st.session_state.df_input),st.session_state.df_input],axis=1)
     except Exception as e:
          st.error(e)
          st.write(traceback.format_exc())

st.dataframe(st.session_state.df_input)

# show prediction
st.subheader('예측결과')
if st.session_state.df_predicted is not None:
     st.dataframe(st.session_state.df_predicted)
     st.download_button(
          label="예측결과를 다운로드 받으세요",
          data=convert_df(st.session_state.df_predicted),
          file_name=str(datetime.datetime.now())[:-7]+'_prediction.csv',
          mime='text/csv',
     )


with st.expander("See explanation"):
     st.header('모델 재학습')

     