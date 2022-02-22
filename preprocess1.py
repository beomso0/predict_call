import pandas as pd
import numpy as np
import joblib
from collections import Counter

joongratio = joblib.load('midcat_ratio.pkl')

# 상품명을 통해서 브랜드명, 가격을 함께 뽑아보아요ㅎㅎ
# 상품명을 key로, 중분류score를 value로 갖는 딕셔너리를 컬럼으로 넣어줍니다.
def makejoong(row):
    valueslist = row['midcat_input']
    prdslist = row['prd_key']
    scoredict = {}
    for i, value in enumerate(valueslist):
        scoredict[prdslist[i]] = joongratio[value]
    return scoredict

# score가 가장 높은 상품의 브랜드를 찾아주는 함수!
def makebrandcol(row): # 가장 큰 값으로 채우기
    brands = row['brand_input']
    prds = row['prd_key']
    scoredict = row['joongscore']
    collist = [] # 컬럼의 내용이 되는 리스트
    if len(scoredict) >= 3: # 상품 개수가 3개 이상
        for prdset in Counter(scoredict).most_common(3): # 각각의 상품에 대해서
            prdnm = prdset[0] # prdnm = 상품명
            index = prds.index(prdnm) # 상품명의 순서 추출
            collist.append(brands[index]) # 해당 순서의 브랜드 리스트에 넣어줍니다
    else: # 상품 개수가 3개 미만
        for prdset in Counter(scoredict).most_common(1):
            prdnm = prdset[0]
            index = prds.index(prdnm)
            collist.append(brands[index])
            collist.append(brands[index])
            collist.append(brands[index])
    return collist

# score가 가장 높은 상품의 가격을 찾아주는 함수!
def makepricecol(row): # 가장 큰 값으로 채우기
    prices = row['price_input']
    prds = row['prd_key']
    scoredict = row['joongscore']
    collist = [] # 컬럼의 내용이 되는 리스트
    if len(scoredict) >= 3: # 상품 개수가 3개 이상
        for prdset in Counter(scoredict).most_common(3):
            prdnm = prdset[0]
            index = prds.index(prdnm)
            collist.append(prices[index])
    else: # 상품 개수가 3개 미만
        for prdset in Counter(scoredict).most_common(1):
            prdnm = prdset[0]
            index = prds.index(prdnm)
            collist.append(prices[index])
            collist.append(prices[index])
            collist.append(prices[index])
    return collist

# score가 가장 높은 상품의 score만 남기는 함수!
def makescorecol(row): # 가장 큰 값으로 채우기
    scoredict = row['joongscore']
    collist = [] # 컬럼의 내용이 되는 리스트
    if len(scoredict) >= 3: # 상품 개수가 3개 이상
        for prdset in Counter(scoredict).most_common(3): # 각각의 상품에 대해서
            prdnm = prdset[0] # prdnm = 상품명
            collist.append(scoredict[prdnm])
    else: # 상품 개수가 3개 미만
        for prdset in Counter(scoredict).most_common(1):
            prdnm = prdset[0]
            collist.append(scoredict[prdnm])
            collist.append(scoredict[prdnm])
            collist.append(scoredict[prdnm])
    return collist

# 전처리 함수 만들기 (main)

def process_input(df):
    # 자료형 변환
    int_cols = [
        'year_input',
        'month_input',
        'time_input',
        'duration_input',
    ]
    df[int_cols] = df[int_cols].astype(int)

    # col name change
    df = df.rename(columns={
        'year_input':'year',
        'month_input':'month',
        'showhost_input': '쇼호스트명',
        'weather_input': '날씨',
        'weekday_input': 'weekday',
        'duration_input': 'duration',
        'live_input': '상품구분',
        'holiday_input':'주말/공휴일'
    })

    # 상품명 대체 key 생성
    df['prd_key'] = df['midcat_input'].apply(lambda x: list(range(len(x))))
    #상품개수
    df['prod_num'] = df['price_input'].apply(lambda x: len(x))

    # df check
    df['prd_info_check'] = df.apply(
        lambda row: len(row['midcat_input']) != row['prod_num'] | len(row['brand_input']) != row['prod_num']
    ,axis=1)
    error_rows = df[df['prd_info_check']]
    if len(error_rows)!=0:
        raise Exception('상품중분류 - 상품 브랜드 - 상품 가격 입력 수가 같지 않은 행이 있습니다.')

    # 쇼호스트명 sorting
    df['쇼호스트명'] = df['쇼호스트명'].apply(lambda x: sorted(x))

    # 상품구분 변환
    df['상품구분'] = df['상품구분'].apply(lambda x: 'LIVE' if x else 'DATA')
    # 주말/공휴일 변환
    df['주말/공휴일'] = df['주말/공휴일'].apply(lambda x: 1 if x else 0)

    #year, month, hour
    df['year'] = df['year']-2020
    df['month' + '_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month' + '_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['hour' + '_sin'] = np.sin(2 * np.pi * df['time_input']/24)
    df['hour' + '_cos'] = np.cos(2 * np.pi * df['time_input']/24)

    #판매가
    df['판매가평균'] = df['price_input'].apply(lambda x: np.mean(x)*1000)
    df['price_max_all'] = df['price_input'].apply(lambda x: max(x)*1000)
    df['price_min_all'] = df['price_input'].apply(lambda x: min(x)*1000)
    
    #한정표현 num
    df['한정표현_num'] = df['expression_input'].apply(lambda x: len(x))

    # sh_num
    df['sh_num'] = df['쇼호스트명'].apply(lambda x: len(x))


    df['joongscore'] = df.apply(lambda row : makejoong(row), axis=1)
    df['brands'] = df.apply(lambda row : makebrandcol(row), axis=1)
    df['prices'] = df.apply(lambda row : makepricecol(row), axis=1)
    df['scores'] = df.apply(lambda row : makescorecol(row), axis=1)
    df['price_max_top3'] = df.prices.apply(lambda row : max(row)*1000)
    df['price_min_top3'] = df.prices.apply(lambda row : min(row)*1000)

    df[['brand1', 'brand2', 'brand3']] = pd.DataFrame(df.brands.tolist(), columns=[['brand1', 'brand2', 'brand3']])
    df[['price1', 'price2', 'price3']] = pd.DataFrame(df.prices.tolist(), columns=[['price1', 'price2', 'price3']])
    df[['score1', 'score2', 'score3']] = pd.DataFrame(df.scores.tolist(), columns=[['score1', 'score2', 'score3']])

    last_cols = ['year', '쇼호스트명', '상품구분', '날씨', 'weekday', 'duration', '판매가평균',
       'price_max_all', 'price_min_all', 'prod_num', '한정표현_num', '주말/공휴일',
       'sh_num', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos','price_max_top3', 'price_min_top3',
       'brand1', 'brand2', 'brand3', 'price1', 'price2', 'price3', 'score1',
       'score2', 'score3']
    
    df = df[last_cols]
    
    return df
  