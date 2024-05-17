from fastbook import *
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from IPython.display import Image, display_svg, SVG
from itertools import chain
import warnings, os

warnings.filterwarnings('ignore')

tab_pandas = None

if os.path.exists("./tab_pandas.pkl"):
    print(f"\n\n>>Found TabularPandas pickle file, loading...")
    tab_pandas = load_pickle("./tab_pandas.pkl")
else:
    #Load training and validation data into PD DF
    df = pd.read_csv('./data/TrainAndValid.csv', low_memory=False)
    df_test = pd.read_csv('./data/Test.csv', low_memory=False)

    print(f"\n\n>>df.columns: \n{df.columns}")
    print(f"\n>>dr.describe(): \n{df.describe()}")

    #Pre-processing data

    print(f"\n\n>>Product size levels: df['ProductSize'].unique()\n{df['ProductSize'].unique()=}")

    sizes = 'Large','Large / Medium','Medium','Small','Mini','Compact'
    print(f"\n>>Set ordered categories for ProductSize: sizes = \n{sizes}")

    #Set ProductSize ordered cats
    df['ProductSize'] = df['ProductSize'].astype('category')
    df['ProductSize'] = df['ProductSize'].cat.set_categories(sizes, ordered=True)

    print(f"\n>>df['ProductSize'] = \n{df['ProductSize']}")

    #Normalize the SalePrice, the target label, to a log value, since it is a monetary value

    dep_var = 'SalePrice'
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




