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


######################################################
#  Combine low frequent Levels of Categorical Data   #
######################################################
# Check the less frequent category within the remaining categorical data
cat_candidates = combined.dtypes[combined.dtypes=="object"].index.values
min_frequency = pd.Series(index=cat_candidates)
for col in cat_candidates:
    min_frequency[col] = combined.loc[:,col].value_counts().min() / combined.shape[0]
def cut_levels(x, threshold, new_value):
    value_counts = x.value_counts()
    labels = value_counts.index[value_counts < threshold]
    x[np.in1d(x, labels)] = new_value
cut_levels(data.class, 30, 'others')

