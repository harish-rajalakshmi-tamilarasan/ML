import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_profiling as pp
import warnings

train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


warnings.filterwarnings('ignore')
pp.ProfileReport(train_df, title = 'Pandas Profiling report of "Train" set', html = {'style':{'full_width': True}})