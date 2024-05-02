from fastai.imports import *
import os, zipfile, kaggle
from numpy import random
from sklearn.model_selection import train_test_split

path=Path('./titanic')
df = pd.read_csv(path/'train.csv')
tst_df = pd.read_csv(path/'test.csv')
modes = df.mode().iloc[0]

print(f"\n\nMissing values: {df.isna().sum().sum()=} / {tst_df.isna().sum().sum()=}")

print(f"\nTraining set modes:\n{modes}")

#Pre-process datasets
def proc_data(df):
    df['Fare'] = df.Fare.fillna(0)
    df.fillna(modes, inplace=True) #Replace NaN in each col with its mode
    df['LogFare'] = np.log1p(df['Fare'])
    df['Embarked'] = pd.Categorical(df.Embarked) #Convert into cat var
    df['Sex'] = pd.Categorical(df.Sex) #Convert into cat var

proc_data(df)
proc_data(tst_df)

print(f"\nTrain data head:\n{df.head()}")
print(f"\nTest data head:\n{tst_df.head()}")

print(f"\n\nMissing values: {df.isna().sum().sum()=} / {tst_df.isna().sum().sum()=}")

cats=["Sex","Embarked"] #categorical features
conts=['Age', 'SibSp', 'Parch', 'LogFare',"Pclass"] #continuous features
dep="Survived" #target feature

trn_df,val_df = train_test_split(df, test_size=0.25)
trn_df[cats] = trn_df[cats].apply(lambda x: x.cat.codes)
val_df[cats] = val_df[cats].apply(lambda x: x.cat.codes)
def xs_y(df):
    xs = df[cats+conts].copy()
    return xs,df[dep] if dep in df else None

trn_xs,trn_y = xs_y(trn_df)
val_xs,val_y = xs_y(val_df) 
def _side_score(side, y):
    tot = side.sum()
    if tot<=1: return 0
    return y[side].std()*tot

def score(col, y, split):
    lhs = col<=split
    return (_side_score(lhs,y) + _side_score(~lhs,y))/len(y)

print(f"\n{score(trn_xs['Sex'], trn_y, 0.5)=}")