import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Use index_col=0 if index column presents in the dataset 
train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)
# Check the first few lines of the DataFrame
train.head()
test.head() 

######################################################
#     Check Data Distribution using Histogram        #
######################################################
# Plot the histogram of a column's data using seaborn
plt.figure(figsize=(20,5))
sns.distplot(train.SalePrice, color="tomato")
plt.title("Target distribution in train")
plt.ylabel("Density")

# Define the combined train/test dataset for Proprocessing
combined = train.drop("SalePrice", axis=1).append(test)


######################################################
#          Transformation of the Data                #
######################################################
# With apply
df['A'].apply(np.log)
# Normalization
df['A'] = (df.A-df.A.mean())/df.A.std()
# Min-Max Normalization
df['A'] = (df.A-df.A.min())/(df.A.max()-df.A.min())

######################################################
#     Compute the Pourcentage of missing data        #
######################################################
## isnull() return a DataFrame of True/False, indicating whether a data is missing
## sum() return a DataFrame of the sum of each column 
## sort_values(by=<column/axis_name>, axis=0, ascending=True, inplace=False)
nan_percentage = combined.isnull().sum().sort_values(ascending=False) / combined.shape[0]
missing_val = nan_percentage[nan_percentage > 0]
# Drop all column that contains too many missing data
to_drop = missing_val[missing_val > 0.1].index.values
combined = combined.drop(to_drop, axis=1)

# Plot the data of missing_val using sns.barplot
plt.figure(figsize=(20,5))
sns.barplot(x=missing_val.index.values, y=missing_val.values * 100, palette="Reds_r");
plt.title("Percentage of missing values in train & test");
plt.ylabel("%");
plt.xticks(rotation=90)


######################################################
#               Handling Missing Data                #
######################################################
# Drop all the columns containing more than 2 NaN
df.dropna(axis=1, how='all', thresh=2, inplace=True)
# Fill the missing data with predefined value at each column
df.fillna({'A':0, 'B':0, 'C':'Unknown'}, inplace=True)
# Fill the missing data with mean/median
df.fillna(df.mean(), inplace=True)
# Replace values
df.replace(-99, np.nan)
# Map some categorical data in a column to some specific values (the rest is filled with NaN)
df['Gender_numeric'] = df.Gender.map(<'Man':1, 'Woman':0>)


######################################################
#          Remove Useless Categorical Data           #
######################################################
# Get categorical data, usually 'object' type
cat_candidates = combined.dtypes[combined.dtypes=="object"].index.values
frequencies = []
# Check the most frequent category within each categorical data
for col in cat_candidates:
    overall_freq = combined.loc[:, col].value_counts().max() / combined.shape[0]
    frequencies.append([col, overall_freq])
# Turn the list 'frequencies' into a DataFrame
frequencies = np.array(frequencies)
freq_df = pd.DataFrame(index=frequencies[:,0], data=frequencies[:,1], columns=["frequency"])
# Change the type of the Data in a DataFrame
freq_df.frequency = freq_df.frequency.astype(np.float)
# Remove the categorical data such that one type of category is too frequent (>90%)
cats_to_drop = freq_df[freq_df.frequency >= 0.9].index.values
combined = combined.drop(cats_to_drop, axis=1)

# Plot the frequencies in ascending order with sns.barplot
sorted_freq = freq_df.frequency.sort_values(ascending=False)
plt.figure(figsize=(20,5))
sns.barplot(x=sorted_freq.index[0:30], y=sorted_freq[0:30].astype(np.float))
plt.xticks(rotation=90)

###             Don't forget numerical categorical data            ###
### Perform the same operations to remove useless categorical data ###
num_candicates = combined.dtypes[combined.dtypes!=object].index.values
# Get the numerical categorical data
next_cat_candidates = [
    "OverallQual", "OverallCond", "MSSubClass", 'BsmtFullBath',
    'BsmtHalfBath','FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'TotRmsAbvGrd', 'Fireplaces', 'GarageCars'
]
frequencies = []
for col in next_cat_candidates:
    overall_freq = combined.loc[:, col].value_counts().max() / combined.shape[0]
    frequencies.append([col, overall_freq])
frequencies = np.array(frequencies)
freq_df = pd.DataFrame(index=frequencies[:,0], data=frequencies[:,1], columns=["frequency"])
sorted_freq = freq_df.frequency.sort_values(ascending=False)
# Plot with barplot to see which categorical data is useless
plt.figure(figsize=(20,5))
sns.barplot(x=sorted_freq.index[0:30], y=sorted_freq[0:30].astype(np.float))
plt.xticks(rotation=90)
# Remove the ones that we do not desire
cats_to_drop = ["KitchenAbvGr", "BsmtHalfBath"]
combined = combined.drop(cats_to_drop, axis=1)


######################################################
#  Combine low frequent Levels of Categorical Data   #
######################################################
# Check the less frequent category within the remaining categorical data
cat_candidates = combined.dtypes[combined.dtypes=="object"].index.values
min_frequency = pd.Series(index=cat_candidates)
for col in cat_candidates:
    min_frequency[col] = combined.loc[:,col].value_counts().min() / combined.shape[0]

# Helper function: combine low frequent level given a threshold
## .isin() returns a DataFrame of booleans showing whether each element in the DataFrame is contained in values.
def cut_levels(x, threshold, new_value):
    value_counts = x.value_counts()
    labels = value_counts.index[value_counts < threshold]
    x[x.isin(labels)] = new_value
cut_levels(data.<col>, 30, 'others')


#######################################################
# Create Numerical Representation of Categorical Data #
#######################################################
# Method 1: (if the data is float but written in string)
## errors='coerce' will replace any non float data by NaN
## errors='ignore' will leave those which are not float type
pd.to_numeric(df.<col>, errors='coerce').fillna(0)

# Method 2: (use category data type)
## Use category type will reduce memory use, make DataFrame faster
df[<col>] = df[<col>].astype('category').cat.codes
## We can also specify in which order each category is encoded
df[<col>] = df[<col>].astype('category', categories=['unknown','man','woman'], ordered=True)

# Method 3: (Encode one category as 1 and the rest 0)
## str.contains(): Return boolean Series or Index based on whether a given pattern or regex is contained within a string of a Series or Index.
df[<col>] = np.where(df[<col>].str.contains('US'))


#######################################################
#         Discretization of Continuous Data           #
#######################################################
# Define the bucket of interval
bucket = [0, 15, 55, 100]
# Relabel the different categories
df.age = pd.cut(df.age, bucket, labels=['child','adult','old'])



#######################################################
#                   Data Selection                    #
#######################################################
# To see visually if a categorical predictor is useful using sns.boxplot
var = 'CentralAir'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
plt.figure(figsize=(20,5))
sns.boxplot(x=var, y="SalePrice", data=data)
plt.axis(ymin=0, ymax=800000)

# To see visually if a numerical predictor is useful using scatter
var = 'YearBuilt'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
plt.scatter(data[var], data["SalePrice"])
sns.regplot(x=var, y='SalePrice', scatter=False, color='Red')
plt.axis(ymin=0, ymax=800000)

# Data correlation
corrmap = train.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corrmap, vmax=0.8, square=True)

k  = 10 # 关系矩阵中将显示10个特征
# corrmat.nlargest(k, 'SalePrice') will return a k*n correlation matrix
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
sns.heatmap(cm, cbar=True, annot=True, \
                 square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


