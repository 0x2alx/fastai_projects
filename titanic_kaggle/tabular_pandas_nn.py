#Based on https://www.kaggle.com/code/jhoward/why-you-should-use-a-framework
import torch, numpy as np, pandas as pd
from torch import tensor
from fastai.data.transforms import RandomSplitter
import torch.nn.functional as F

pd.options.display.float_format = '{:.2f}'.format

#Load csv training set in pd df
pd_dataframe = pd.read_csv('titanic/train.csv')
print(f"\nLoaded training data:\n{pd_dataframe}")

#Pre-processing of features
def add_features(df):
    df['LogFare'] = np.log1p(df['Fare']) #log of monetary value
    df['Deck'] = df.Cabin.str[0].map(dict(A="ABC", B="ABC", C="ABC", D="DE", E="DE", F="FG", G="FG"))
    df['Family'] = df.SibSp+df.Parch
    df['Alone'] = df.Family==0
    df['TicketFreq'] = df.groupby('Ticket')['Ticket'].transform('count')
    df['Title'] = df.Name.str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    df['Title'] = df.Title.map(dict(Mr="Mr",Miss="Miss",Mrs="Mrs",Master="Master"))
add_features(pd_dataframe)
print(f"\nPre-processing complete: {pd_dataframe.columns=}")
print(f"\nDF describe:\n{pd_dataframe.describe()}")
#Split data intro traning and validation sets
splits = RandomSplitter(seed=42)(pd_dataframe)
print(f"\n Data split: Training samples = {len(splits[0])} / Validation samples = {len(splits[1])}")
