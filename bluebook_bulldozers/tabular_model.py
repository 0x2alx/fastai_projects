from fastbook import *
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from IPython.display import Image, display_svg, SVG
from itertools import chain
import warnings, os

warnings.filterwarnings('ignore')

#Load training and validation data into PD DF
df = pd.read_csv('./data/TrainAndValid.csv', low_memory=False)
df_test = pd.read_csv('./data/Test.csv', low_memory=False)

print(f"\n\n>>df.columns: \n{df.columns}")
print(f"\n>>dr.describe(): \n{df.describe()}")

#Pre-processing data

print(f"\n\n>>Product size levels: df['ProductSize'].unique()\n{df['ProductSize'].unique()=}")

sizes = 'Large','Large / Medium','Medium','Small','Mini','Compact'
print(f"\n>>Set ordered categories for ProductSize: sizes = \n{sizes}")
dep_var = 'SalePrice'

#Set ProductSize ordered cats
df['ProductSize'] = df['ProductSize'].astype('category')
df['ProductSize'] = df['ProductSize'].cat.set_categories(sizes, ordered=True)

print(f"\n>>df['ProductSize'] = \n{df['ProductSize']}")

#Normalize the SalePrice, the target label, to a log value, since it is a monetary value

df[dep_var] = np.log(df[dep_var])

print(f"\n\n>>Normalized/log the target label: df[dep_var] = \n{df[dep_var]}")


#Add fast.ai date features
df = add_datepart(df, 'saledate')
df_test = add_datepart(df_test, 'saledate')

new_cols = ', '.join(o for o in df.columns if o.startswith('sale'))
print(f"\n\n>>Newly added date cols:\n{new_cols}")



##Convert to TabularPandas to access fast transforms
#Tabular pre-processing
procs = [Categorify,FillMissing]

#Split into training and validation
cond = (df.saleYear<2011) | ((df.saleYear==2011) & (df.saleMonth<10))
train_idx = np.where(cond)[0]
valid_idx = np.where(~cond)[0]
print(f"\n\n>>Train data set size: {len(train_idx)}")
print(f">>Valid data set size: {len(valid_idx)}")
splits = (list(train_idx),list(valid_idx))

#Define continuous and categorical variables
cont, cat = cont_cat_split(df, 1, dep_var=dep_var)
print(f"\n\n>>Continuous columns:\n{cont}")
print(f"\n>>Categorical columns:\n{cat}")

#Create the TabularPandas object
tab_pandas = TabularPandas(df, procs=procs, cat_names=cat, cont_names=cont, y_names=dep_var, splits=splits)
print(f"\n\n>>{len(tab_pandas.train)=}")
print(f">>{len(tab_pandas.valid)=}")

save_pickle("./tab_pandas.pkl",tab_pandas)

#Get training and validation inputs and targets
train_xs, train_y = tab_pandas.train.xs, tab_pandas.train.y
valid_xs, valid_y = tab_pandas.valid.xs, tab_pandas.valid.y

#Replace all missing/wrong dates with defautl value
train_xs.loc[train_xs['YearMade']<1900, 'YearMade'] = 1950
valid_xs.loc[valid_xs['YearMade']<1900, 'YearMade'] = 1950

#Create RMSE loss function
def r_mse(pred,y):
    return round(math.sqrt(((pred-y)**2).mean()), 6)
def m_rmse(m, xs, y):
    return r_mse(m.predict(xs),y)

print(f"\n\n>>train_xs:{len(train_xs)}")
print(f">>valid_xs:{len(valid_xs)}")

#Create RandomForests function
def rf(xs, y, n_trees=40, max_samples=200_000, max_features=0.5, min_sample_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_trees, 
            max_samples=max_samples, max_features=max_features, 
            min_samples_leaf=min_sample_leaf, oob_score=True).fit(xs, y)

#Create an RF tree
rf_model = rf(train_xs, train_y)

#Print training and validation set losses
print(f"\n\n>>Training loss: {m_rmse(rf_model, train_xs, train_y)}")
print(f">>Validation loss: {m_rmse(rf_model, valid_xs, valid_y)}")

#Get validation set predictions
preds = np.stack([t.predict(valid_xs) for t in rf_model.estimators_])
print(f"\n\n>>Validation set predictions shape:\n{preds.shape}")

#Print the validation set loss per number of trees in the RF
print(f"\n\n>>RMSE per number of trees")
print("\n".join([f"{i+1}: {r_mse(preds[:i+1].mean(0), valid_y):.4f}" for i in range(5)]))
print("...")
print("\n".join([f"{i+1}: {r_mse(preds[:i+1].mean(0), valid_y):.4f}" for i in range(35,40)]))

#Define feature importance
def rf_feat_importance(modell, df):
    return pd.DataFrame({'cols':df.columns, 'importance':modell.feature_importances_}).sort_values('importance',ascending=False)

feat_imps = rf_feat_importance(rf_model,train_xs)
print(f"\n\n>>Feature importance:\n{feat_imps}")

#Eliminate features with lower importance to simplify the model
features_to_keep = feat_imps[feat_imps.importance>0.005].cols
print(f"\nEliminating low importance features to simplify model...")
print(f">>Keeping {len(features_to_keep)} out of {len(tab_pandas.items.columns)}")

train_xs_trimmed = train_xs[features_to_keep]
valid_xs_trimmed = valid_xs[features_to_keep]

#Create a new tree with trimmed cols
tab_model_trimmed = rf(train_xs_trimmed, train_y)
print(f"\n\n>>Loss with full model:\nTrain = {m_rmse(rf_model, train_xs, train_y)}\nValidation = {m_rmse(rf_model, valid_xs, valid_y)}")
print(f"\n>>Loss with trimmed model:\nTrain = {m_rmse(tab_model_trimmed,train_xs_trimmed,train_y)}\nValidation = {m_rmse(tab_model_trimmed,valid_xs_trimmed,valid_y)}")

#Create function that returns oob score
def get_oob(df):
    mod = RandomForestRegressor(n_estimators=40, min_samples_leaf=15,
                max_samples=50000, max_features=0.5, n_jobs=-1, oob_score=True)
    mod.fit(df, train_y)
    return mod.oob_score_

print(f"\n\n>>OOB score for trimmed model: {get_oob(train_xs_trimmed):.4f}")
print(f"OOB score after dropping specific columns:")
oob_scores_cols = {c:get_oob(train_xs_trimmed.drop(c, axis=1)) for c in (
    'saleYear', 'saleElapsed', 'ProductGroupDesc','ProductGroup',
    'fiModelDesc', 'fiBaseModel',
    'Hydraulics_Flow','Grouser_Tracks', 'Coupler_System')}
print("\n".join([f"{key}: {value:.4f}" for key, value in oob_scores_cols.items()]))

mult_cols_to_drop = ['saleYear', 'ProductGroupDesc', 'fiBaseModel', 'Grouser_Tracks']

print(f"\nOOB score after dropping multiple cols: {mult_cols_to_drop}")
print(f"{get_oob(train_xs_trimmed.drop(mult_cols_to_drop, axis=1))}")

#Create new train and valid data sets after dropping selected columns
train_xs_final = train_xs_trimmed.drop(mult_cols_to_drop,axis=1)
valid_xs_final = valid_xs_trimmed.drop(mult_cols_to_drop,axis=1)

print(f"\n\nSaving final training and validation data frames...")
save_pickle("./train_xs_final.pkl",train_xs_final)
save_pickle("./valid_xs_final.pkl",valid_xs_final)


#Check accuracy/loss on final data sets
final_model = rf(train_xs_final, train_y)
print(f"\n\n>>Final training loss: {m_rmse(final_model, train_xs_final, train_y)}")
print(f">>Final validation loss: {m_rmse(final_model, valid_xs_final, valid_y)}")


### Compare with a neural net model

df_nn = pd.read_csv('./data/TrainAndValid.csv', low_memory=False)
df_nn['ProductSize'] = df_nn['ProductSize'].astype('category')
df_nn['ProductSize'] = df_nn['ProductSize'].cat.set_categories(sizes, ordered=True)
df_nn[dep_var] = np.log(df_nn[dep_var])
df_nn = add_datepart(df_nn, 'saledate')

df_nn_final = df_nn[list(train_xs_final.columns) + [dep_var]]

cont_nn, cat_nn = cont_cat_split(df_nn_final, max_card=9000, dep_var=dep_var)

#Duplicate info
cat_nn.remove('fiModelDescriptor')

print(f"{df_nn_final[cat_nn].nunique()}")

procs_nn = [Categorify, FillMissing, Normalize]
tab_pandas_nn = TabularPandas(df_nn_final, procs_nn, cat_nn, cont_nn, splits=splits, y_names=dep_var)

dls = tab_pandas_nn.dataloaders(1024)

learn = tabular_learner(dls, y_range=(8,12), layers=[500,250], n_out=1, loss_func=F.mse_loss)

print(f"\n\nLR finder:\n{learn.lr_find()}\n\n")
learn.fit_one_cycle(5, 1e-2)

preds,targs = learn.get_preds()
print(f"\nLOSS: {r_mse(preds,targs)}")

learn.save('neural_network_model')

#Ensembling the RF and NN model predictions
rf_preds = final_model.predict(valid_xs_final)
end_preds = (to_np(preds.squeeze()) + rf_preds)/2

print(f"\n\n>>Ensemble of RF and NN LOSS: {r_mse(end_preds,valid_y)}")
