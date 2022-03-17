import numpy as np
import pandas as pd
import copy
import datetime

def process_backup(backup_csv):
    try:
        try:
            df = pd.read_csv(backup_csv,encoding='cp949')
        except:
            df = pd.read_csv(backup_csv)
    except:
        try:
            df = pd.read_excel(backup_csv,encoding='cp949')
        except:
            df = pd.read_excel(backup_csv)

    try:
        df['date_input'] = df['date_input'].apply(lambda x: datetime.date(int(x[:4]), int(x[5:7]),int(x[8:])))
    except:
        df['date_input'] = df['date_input'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]),int(str(x)[6:])))

    for c in ['showhost_input',	'expression_input', 'midcat_input',	'brand_input']:
        df[c] = df[c].apply(lambda x: sorted(eval(x)))
    return df

def process_input(input_df,score_lists,cat_encoder):
    df = copy.deepcopy(input_df)
    
    df.columns = [col[:-6] for col in df.columns]
    df = df.replace({True:1, False:0})
    df['pgm'] = df['pgm'].astype('str')
    
    df['showhost_num'] = df['showhost'].apply(lambda x: len(x))
    df['midcat_num'] = df['midcat'].apply(lambda x: len(x))
    df['brand_num'] = df['brand'].apply(lambda x: len(x))
    df['expression_num'] = df['expression'].apply(lambda x: len(x))
     
    df['showhost'] = df['showhost'].apply(lambda x: str(sorted(x))[2:-2].replace("'",''))
    
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['weekday'] = df['date'].apply(lambda x: x.weekday())

    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['weekday_sin'] = np.sin(2*np.pi*df['weekday']/7)
    df['weekday_cos'] = np.cos(2*np.pi*df['weekday']/7)
    df['start_time_sin'] = np.sin(2*np.pi*df['start_time']/24)
    df['start_time_cos'] = np.cos(2*np.pi*df['start_time']/24)
    df['end_time_sin'] = np.sin(2*np.pi*df['end_time']/24)
    df['end_time_cos'] = np.cos(2*np.pi*df['end_time']/24)
    
    df['price_min'] = df['price_min']*1000
    df['price_max'] = df['price_max']*1000
    df['price_mean'] = df['price_mean']*1000
    
    
    def make_top_col(brand_list,score_df,col_name):
        temp = score_df
        for brand in set(brand_list):
            if brand not in temp[col_name]:
                new = pd.DataFrame([[brand,0,0,score_df[score_df.columns[-1]].mean()]], columns=score_df.columns)
                temp = temp.append(new,ignore_index=True)
            else:
                continue
        
        temp = temp.sort_values(by=[temp.columns[-1],'방송등장횟수'],ascending=False).reset_index(drop=True)
        temp2 = list(temp[temp[col_name].isin(list(set(brand_list)))][col_name])
        return temp2
    
    
    df['top_midcat'] = df['midcat'].apply(lambda x: make_top_col(x,score_lists[0],'상품중분류명'))
    df['midcat1'] = df['top_midcat'].apply(lambda x: x[0])
    df['midcat2'] = df['top_midcat'].apply(lambda x: x[1] if len(x)>=2 else x[0])
    df['midcat3'] = df['top_midcat'].apply(lambda x: x[2] if len(x)>=3 else x[0])
    df['top_brand'] = df['brand'].apply(lambda x: make_top_col(x,score_lists[1],'브랜드명'))
    df['brand1'] = df['top_brand'].apply(lambda x: x[0])
    df['brand2'] = df['top_brand'].apply(lambda x: x[1] if len(x)>=2 else x[0])
    df['brand3'] = df['top_brand'].apply(lambda x: x[2] if len(x)>=3 else x[0])
    df['top_expression'] = df['expression'].apply(lambda x: make_top_col(x,score_lists[2],'한정표현구분'))
    df['expression1'] = df['top_expression'].apply(lambda x: x[0])
    df['expression2'] = df['top_expression'].apply(lambda x: x[1] if len(x)>=2 else x[0])
    df['expression3'] = df['top_expression'].apply(lambda x: x[2] if len(x)>=3 else x[0])
    
    drop_cols = ['date','year','month','weekday','start_time','end_time',
                 'midcat','brand','expression','top_midcat','top_brand','top_expression']
    df = df.drop(drop_cols, axis=1)
    
    df = df[['pgm','live','showhost','duration','temp','holiday',
             'month_sin','month_cos','weekday_sin','weekday_cos',
             'start_time_sin','start_time_cos','end_time_sin','end_time_cos','day',
             'showhost_num','midcat_num','brand_num','expression_num','product_num',
             'price_min','price_max','price_mean',
             'midcat1','midcat2','midcat3','brand1','brand2','brand3','expression1','expression2','expression3']]
    
    df.columns = ['PGM코드','상품구분','쇼호스트명','방송길이','기온','주말제외공휴일',
                  'MONTH_SIN','MONTH_COS','WEEKDAY_SIN','WEEKDAY_COS','HOUR_START_SIN','HOUR_START_COS','HOUR_END_SIN','HOUR_END_COS','DAY',
                  '쇼호스트_num','상품중분류_num','브랜드_num','한정표현_num','판매상품개수',
                  '상품판매가_min','상품판매가_max','상품판매가_mean',
                  '상품중분류1','상품중분류2','상품중분류3','브랜드1','브랜드2','브랜드3','한정표현1','한정표현2','한정표현3']
    
    df = cat_encoder.transform(df)
    
    return df