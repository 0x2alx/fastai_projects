from fastai.imports import *
import os, zipfile, kaggle
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier

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

def min_col(df, nm):
    col,y = df[nm],df[dep]
    unq = col.dropna().unique()
    scores = np.array([score(col, y, o) for o in unq if not np.isnan(o)])
    idx = scores.argmin()
    return unq[idx],scores[idx]

cols = cats+conts
print(f"\n\nBest splits (based on scores) per feature:\n")
dictt = {o:min_col(trn_df, o) for o in cols}

#For each feature, print the best spliting value, as well as the score according to that split
for key in dictt:
    print(f"{key}: ({dictt[key][0]:.3f}, {dictt[key][1]:.3f})")

#Function to create a single decision tree
def get_tree(prop=0.75):
    n = len(trn_y)
    idxs = random.choice(n, int(n*prop))
    return DecisionTreeClassifier(min_samples_leaf=5).fit(trn_xs.iloc[idxs], trn_y.iloc[idxs])

#Create a MANUAL RandomForest (ensemble of 100 trees)
trees = [get_tree() for t in range(100)]

#Calculate all probabilities and their average
all_probs = [t.predict(val_xs) for t in trees]
avg_probs = np.stack(all_probs).mean(0)

print(f"\n\nMAE for manual RandomForests: {mean_absolute_error(val_y, avg_probs):.2f}")

#Create an RF from sklearn
rf = RandomForestClassifier(100, min_samples_leaf=5)
rf.fit(trn_xs, trn_y);
print(f"\nMAE for SKLEARN RandomForests: {mean_absolute_error(val_y, rf.predict(val_xs)):.2f}")

print(f"\n\nFeature importance:")
for name, importance in zip(trn_xs.columns, rf.feature_importances_):
    print(f"{name}: {importance:.4f}")