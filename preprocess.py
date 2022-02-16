#%%
import pandas as pd
import numpy as np
import joblib
from pyLDAvis import display
import matplotlib.pyplot as plt

test_df = joblib.load('./sh_preprocess/process_test.pkl')
midcat_ratio = pd.read_csv('./sh_preprocess/midcat_ratio.csv')
print(test_df)



'''
<input df structure>
[
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
]

'''