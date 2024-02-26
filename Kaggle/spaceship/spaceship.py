import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder   
from scipy.stats import chi2_contingency

df_train = pd.read_csv(r'D:\\DS\\Kaggle\\spaceship\\train.csv')
df_test = pd.read_csv(r'D:\\DS\\Kaggle\\spaceship\\test.csv')
df_merged = pd.concat([df_train, df_test], sort=False).drop(['Transported'], axis=1)

df_merged['group_id'] = df_merged['PassengerId'].str.split('_').str[0]
df_merged['individual_number'] = df_merged['PassengerId'].str.split('_').str[1]
df_merged['lastName'] = df_merged['Name'].str.split(' ').str[1]

# Convert the boolean and categorical columns to string
#df_merged['CryoSleep'] = df_merged['CryoSleep'].astype(str)
#df_merged['Destination'] = df_merged['Destination'].astype(str)



def fillHomePlanet(name_group):
    if name_group['HomePlanet'].notnull().any():
        most_common = name_group['HomePlanet'].value_counts().idxmax()
        name_group['HomePlanet'] = name_group['HomePlanet'].fillna(most_common)
    
    return name_group

most_common_home_planet = df_merged['HomePlanet'].value_counts().idxmax()
groupid_groups = df_merged.groupby(['group_id','lastName'])
df_merged_grouped = groupid_groups.apply(fillHomePlanet)

mask = df_merged['group_id'].isna() | df_merged['lastName'].isna()
df_merged_nan_groups = df_merged[mask].copy()
df_merged = pd.concat([df_merged_grouped, df_merged_nan_groups])

df_merged['HomePlanet'] = df_merged['HomePlanet'].fillna(most_common_home_planet)

def fillLastName(name_group):
    if name_group['lastName'].notnull().any():
        most_common = name_group['lastName'].value_counts().idxmax()
        name_group['lastName'] = name_group['lastName'].fillna(most_common)
    return name_group

groupid_planet_grp = df_merged.groupby(['group_id','HomePlanet'])
df_merged = groupid_planet_grp.apply(fillLastName).reset_index(drop=True)
df_merged['lastName'] = df_merged['lastName'].fillna('Alone')

def fillCryoSleep(id_group):
    if id_group['CryoSleep'].notnull().any():
        most_common = id_group['CryoSleep'].value_counts().idxmax()
        id_group['CryoSleep'] = id_group['CryoSleep'].fillna(most_common)
    else:
        # Handle the case where all values are NaN, choose a default value or strategy
        id_group['CryoSleep'] = id_group['CryoSleep'].fillna(False) # Example: default to False
    return id_group

groupid_grp = df_merged.groupby('group_id')
df_merged = groupid_grp.apply(fillCryoSleep).reset_index(drop=True)

def fillDestination(id_group):
    if id_group['Destination'].notnull().any():
        most_common = id_group['Destination'].value_counts().idxmax()
        id_group['Destination'] = id_group['Destination'].fillna(most_common)
   
    return id_group

most_common_destination = df_merged['Destination'].value_counts().idxmax()
groupid_grp = df_merged.groupby('group_id')
df_merged = groupid_grp.apply(fillDestination).reset_index(drop=True)
df_merged['Destination'] = df_merged['Destination'].fillna(most_common_destination)

count_df = df_train.groupby(['Age','Transported']).size().unstack(fill_value=0)

# Plotting
count_df.plot(kind='bar', stacked=True)
plt.xlabel('VIP')
plt.ylabel('Transported')
plt.title('Stacked Bar Chart of VIP and Transported')
plt.show()

df_merged['total_price']=df_merged['Spa'].fillna(0)+df_merged['RoomService'].fillna(0)+df_merged['FoodCourt'].fillna(0)+df_merged['ShoppingMall'].fillna(0)

print(df_merged.isnull().sum())

def convertTextToNumbers():
    df_merged['Destination'] = df_merged['Destination'].replace('TRAPPIST-1e', 0).replace('55 Cancri e', 1).replace('PSO J318.5-22', 2)
    df_merged['HomePlanet'] = df_merged['HomePlanet'].replace('Earth', 0).replace('Europa', 1).replace('Mars', 2)

    
#convertTextToNumbers()



def print_info(dataset):
    print(dataset.info())
    print(dataset.describe())
    print(dataset.head(10))
    print(dataset.tail(10))
    print(dataset.shape)
    print(dataset.columns)
    print(dataset.isnull().sum())


def bar_chart_stacked(dataset, feature, stacked=True):
    transported = dataset[dataset['Transported']==1][feature].value_counts()
    not_transported = dataset[dataset['Transported']==0][feature].value_counts()
    df_transported = pd.DataFrame([transported,not_transported])
    df_transported.index = ['Transported','Not Transported']
    ax = df_transported.plot(kind='bar',stacked=stacked, figsize=(10,5))  
    plt.show()

def stacked_bar(dataset, feature1, feature2):
    df = pd.crosstab(dataset[feature1],dataset[feature2])
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    plt.show()

def scatter(dataset, feature1, feature2):
    plt.scatter(dataset[feature1],dataset[feature2], alpha=0.5)
    plt.show()


#print(df_merged['Destination'].value_counts())
#corr_matrix = df_train.corr()
#print(corr_matrix["Transported"].sort_values(ascending=False))

#print_info(df_merged)
#stacked_bar(df_train,"HomePlanet","Destination")
