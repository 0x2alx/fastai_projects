#Based on https://www.kaggle.com/code/alexm42/titanic-advanced-feature-engineering-tutorial
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import string

df_train = pd.read_csv('titanic/train.csv')
df_test = pd.read_csv('titanic/test.csv')
df_all = pd.concat([df_train, df_test], sort=True).reset_index(drop=True)
dfs = [df_train, df_test]
df_train.df_name = "--> Training DF"
df_test.df_name = "--> Test DF"

print('Number of Training Examples = {}'.format(df_train.shape[0]))
print('Number of Test Examples = {}\n'.format(df_test.shape[0]))
print('Training X Shape = {}'.format(df_train.shape))
print('Training y Shape = {}\n'.format(df_train['Survived'].shape[0]))
print('Test X Shape = {}'.format(df_test.shape))
print('Test y Shape = {}\n'.format(df_test.shape[0]))
print(f"Training cols: {df_train.columns}")
print(f"Test cols: {df_test.columns}")

print(f"\n\n")
print(f"--> TRAIN DF:")
df_train.info()
print(df_train)
print(f"\n--> TEST DF:")
df_test.info()
print(df_test)

#Display how many missing values there are per feature
def display_missing(df):    
    for col in df.columns.tolist():          
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')
for df in dfs:
    print('{}'.format(df.df_name))
    display_missing(df)

#Display correlations between Age and other continuous features
df_all_corr = df_all.select_dtypes(include='number').corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
print(df_all_corr[df_all_corr['Feature 1'] == 'Age'])

#Fill Age missing values with median for matching PClass
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

#Fill missing embaarked values (only 2)
df_all['Embarked'] = df_all['Embarked'].fillna('S')

#Calulate median fare and replace missing values
med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
df_all['Fare'] = df_all['Fare'].fillna(med_fare)

# Passenger in the T deck is changed to A
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
idx = df_all[df_all['Deck'] == 'T'].index
df_all.loc[idx, 'Deck'] = 'A'

#Group deck values into groups with similar survivability
df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')
#Drop cabin for deck
df_all.drop(['Cabin'], inplace=True, axis=1)

#Split up the processed dataset
df_train, df_test = df_all.loc[:890], df_all.loc[891:].drop(['Survived'], axis=1)
dfs = [df_train, df_test]

print(f"\n\nTraining survival rate: {(df_train['Survived'].value_counts()[1]/df_train.shape[0])*100:.2f}%")
df_all = pd.concat([df_train, df_test], sort=True).reset_index(drop=True)

#Split Fare into 13 equal sized quantiles
df_all['Fare'] = pd.qcut(df_all['Fare'], 13)

#Split Age into 10 quantiles
df_all['Age'] = pd.qcut(df_all['Age'], 10)

#Create new feature family_size
df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1

#Group tickets by frequency to reduce cardinality
df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')

#Extract Title feature from Name and group them
df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')

#Create is_married feature
df_all['Is_Married'] = 0
df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1

#Function to extract family name from Name feature
def extract_surname(data):    
    
    families = []
    
    for i in range(len(data)):        
        name = data.iloc[i]

        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        
        for c in string.punctuation:
            family = family.replace(c, '').strip()
            
        families.append(family)
            
    return families

#Create family name feature
df_all['Family'] = extract_surname(df_all['Name'])
df_train = df_all.loc[:890]
df_test = df_all.loc[891:]  
dfs = [df_train, df_test]