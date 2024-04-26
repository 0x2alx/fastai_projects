import torch, numpy as np, pandas as pd
from torch import tensor
from fastai.data.transforms import RandomSplitter
import torch.nn.functional as F

#Based on https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch

torch.manual_seed(42)

#Load csv training set in pd df
pd_dataframe = pd.read_csv('titanic/train.csv')
print(f"\nLoaded training data:\n{pd_dataframe}")

#Replace all NaN values with mode of the respective col
pd_dataframe.fillna(pd_dataframe.mode().iloc[0], inplace=True)
print(f"\nMaking sure we dont have any NaN values left in the data:\n{pd_dataframe.isna().sum()}")

#Normalize the monetary column (and add 1 because log(0) is infinite)
pd_dataframe['LogFare'] = np.log(pd_dataframe['Fare']+1)

#Create dummy columns for categorical columns
pd_dataframe = pd.get_dummies(pd_dataframe, columns=["Sex","Pclass","Embarked"])
print(f"\nColumns:\n{pd_dataframe.columns}")

#We only consider the continuous variables of interest and the dummy variables created above
dummy_cols = ['Sex_male', 'Sex_female', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
input_cols = ['Age', 'SibSp', 'Parch', 'LogFare']+dummy_cols
pd_dataframe[dummy_cols] = pd_dataframe[dummy_cols].astype(float)
print(f"\n{pd_dataframe[input_cols]}")

#Create the target/dependent_var training tensor
training_target_full_tensor = tensor(pd_dataframe.Survived)
#Create the input/independent_var training tensor
training_input_full_tensor = torch.tensor(pd_dataframe[input_cols].values, dtype=torch.float)
#Normalize our coumns (divide each value by the max of its column)
max_col_vals,_ = training_input_full_tensor.max(dim=0)
training_input_full_tensor = training_input_full_tensor / max_col_vals
print(f"\n{training_input_full_tensor.shape=}")
print(f"\n{training_input_full_tensor=}")
print(f"\n{training_target_full_tensor.shape=}")
print(f"\n{training_target_full_tensor=}")

#Number of coefficients/weights/dataset features
nb_coefficients = training_input_full_tensor.shape[1]
print(f"\n{nb_coefficients=}")

#Split Training dataset intor training/validation
trn_split,val_split=RandomSplitter(seed=42)(pd_dataframe)
training_input_tensor, validation_input_tensor = training_input_full_tensor[trn_split], training_input_full_tensor[val_split]
training_target_tensor, validation_target_tensor = training_target_full_tensor[trn_split], training_target_full_tensor[val_split]
training_target_tensor = training_target_tensor[:,None]
validation_target_tensor = validation_target_tensor[:,None]
print(f"\n{training_input_tensor.shape=}")
print(f"{validation_input_tensor.shape=}")
print(f"{training_target_tensor.shape=}")
print(f"{validation_target_tensor.shape=}")

#Initialize random weights for input layer and hidden layers
def init_coeffs(): 
    hidden_layers_out = [10, 10]
    layer_sizes = [nb_coefficients] + hidden_layers_out + [1]
    n = len(layer_sizes)
    layers = [(torch.rand(layer_sizes[i], layer_sizes[i+1])-0.3)/layer_sizes[i+1]*4 for i in range(n-1)]
    consts = [(torch.rand(1)[0]-0.5)*0.1 for i in range(n-1)]
    for l in layers+consts: l.requires_grad_()
    return layers,consts

#Function to calculate predictions from input and coefficients
def calc_preds(coeffs, indeps): 
    layers,consts = coeffs
    n = len(layers)
    res = indeps
    for i,l in enumerate(layers):
        res = res@l + consts[i]
        if i!=n-1: res = F.relu(res)
    return torch.sigmoid(res)

#Function to calculate avg loss between predictions and targets (Mean Absolute Error)
def calc_loss(coeffs, indeps, deps): 
    return torch.abs(calc_preds(coeffs, indeps)-deps).mean()

#Do gradient descent on our weights towards lower loss
def update_coeffs(coeffs, lr):
    layers,consts = coeffs
    for layer in layers+consts:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()

#A function combining one epoch of training
def one_epoch(coeffs, lr):
    loss = calc_loss(coeffs, training_input_tensor, training_target_tensor)
    loss.backward()
    with torch.no_grad(): update_coeffs(coeffs, lr)
    print(f"\nloss = {loss:.3f}", end="; ")

#Train model for multiple epochs
def train_model(epochs=30, lr=0.001):
    torch.manual_seed(442)
    coeffs = init_coeffs()
    for i in range(epochs): one_epoch(coeffs, lr=lr)
    return coeffs    

coeffs = train_model(60, lr=4)
#print(f"\n{coeffs=}")

def calc_accuracy(coeffs): 
    return (validation_target_tensor.bool()==(calc_preds(coeffs, validation_input_tensor)>0.5)).float().mean()

print(f"\nAccuracy = {calc_accuracy(coeffs)}")
