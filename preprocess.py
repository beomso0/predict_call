#%%
import numpy as np
import pandas as pd

def process_input(input_df, score_lists, cat_encoder):    
    df.columns = [col[:-6] for col in df.columns]
    df = df.replace({True:1, False:0})
    df['pgm'] = df['pgm'].astype('str')
    df['showhost'] = df['showhost'].apply(lambda x: str(x)[2:-2].replace("'",''))
    
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
    
    df['price_min'] = df['price'].apply(lambda x: min(x))
    df['price_max'] = df['price'].apply(lambda x: max(x))
    df['price_mean'] = df['price'].apply(lambda x: np.mean(x))
    
    
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
    
    
    df['top_midcat'] = df['midcat'].apply(lambda a: make_top_col(a,score_lists[0],'상품중분류명'))
    df['midcat1'] = df['top_midcat'].apply(lambda a: a[0])
    df['midcat2'] = df['top_midcat'].apply(lambda a: a[1] if len(a)>=2 else a[0])
    df['midcat3'] = df['top_midcat'].apply(lambda a: a[2] if len(a)>=3 else a[0])
    df['top_brand'] = df['brand'].apply(lambda a: make_top_col(a,score_lists[1],'브랜드명'))
    df['brand1'] = df['top_brand'].apply(lambda a: a[0])
    df['brand2'] = df['top_brand'].apply(lambda a: a[1] if len(a)>=2 else a[0])
    df['brand3'] = df['top_brand'].apply(lambda a: a[2] if len(a)>=3 else a[0])
    df['top_expression'] = df['expression'].apply(lambda a: make_top_col(a,score_lists[2],'한정표현구분'))
    df['expression1'] = df['top_expression'].apply(lambda a: a[0])
    df['expression2'] = df['top_expression'].apply(lambda a: a[1] if len(a)>=2 else a[0])
    df['expression3'] = df['top_expression'].apply(lambda a: a[2] if len(a)>=3 else a[0])
    
    drop_cols = ['date','year','month','weekday','start_time','end_time',
                 'midcat','brand','expression','price','top_midcat','top_brand','top_expression']
    df = df.drop(drop_cols, axis=1)
    
    df = cat_encoder.transform(df)
    
    return df