from fastbook import *
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from IPython.display import Image, display_svg, SVG
import warnings

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


