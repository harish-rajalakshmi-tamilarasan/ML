import pandas as pd

df = pd.read_csv(r'D:\DS\survey_results_public.csv')
print(df[['Respondent','Age']])