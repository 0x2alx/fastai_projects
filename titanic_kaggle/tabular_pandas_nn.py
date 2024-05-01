#Based on https://www.kaggle.com/code/jhoward/why-you-should-use-a-framework
import torch, numpy as np, pandas as pd
from torch import tensor
from fastai.data.transforms import RandomSplitter
from fastai.tabular.all import *
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

#Create TabularPandas dataloaders for training and validation set
dls = TabularPandas(
    pd_dataframe, splits=splits,
    procs = [Categorify, FillMissing, Normalize], #Turn strs into categories, fill NaN with median, Normalize all cols
    cat_names=["Sex","Pclass","Embarked","Deck", "Title"],
    cont_names=['Age', 'SibSp', 'Parch', 'LogFare', 'Alone', 'TicketFreq', 'Family'],
    y_names="Survived", y_block = CategoryBlock(), #Specify the type of dep variable, it is a categorical target, so building a classification model (not regression one)
).dataloaders(path=".")
print(f"\n TabularPandas dls: \n{dls}")

#Data + model = Learner (with 2 hidden layers)
learn = tabular_learner(dls, metrics=accuracy, layers=[10,10])
print(f"\n tabular_learner learn: \n{learn}")

#Find ideal learning rate
slide_lr, valley_lr = learn.lr_find(suggest_funcs=(slide, valley))
print(f"\n slide_lr = {slide_lr:.4f} \n valley_lr = {valley_lr:.4f}")

#Train the learner with appropriate LR
learn.fit(16, lr=0.03)

#Export the trained model to file
learn.export("tabular_pandas_nn.pkl")


### Get predictions on test set
test_pd_dataframe = pd.read_csv("titanic/test.csv")
test_pd_dataframe['Fare'] = test_pd_dataframe.Fare.fillna(0) #Fill Fare missing values with 0

#Add extra features like for training set
add_features(test_pd_dataframe)

#Add test dataframe to our learner's dls so it can process it with the same transformations as the training/validation sets
test_dataloader = learn.dls.test_dl(test_pd_dataframe)

#Get predictions for the test dataset
preds, _ = learn.get_preds(dl=test_dataloader)
print(f"\nPredictions shape = {preds.shape}")
print(f"\nSample predictions: \n{preds[0:5]}")

#Create new col in our test dataframe with the Survived predictions
test_pd_dataframe['Survived'] = (preds[:,1]>0.5).int()

#Create new scaled down data frame
sub_test_pd_dataframe = test_pd_dataframe[['PassengerId','Survived']]
print(f"\n Test set survived = {(sub_test_pd_dataframe['Survived'] == 1).sum()}")
print(f" Test set did not survive = {(sub_test_pd_dataframe['Survived'] == 0).sum()}")

#submission_csv = sub_test_pd_dataframe.to_csv(index=False)
