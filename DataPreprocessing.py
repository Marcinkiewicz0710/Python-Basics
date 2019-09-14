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

# Plot the histogram of a column's data using seaborn
plt.figure(figsize=(20,5))
sns.distplot(train.SalePrice, color="tomato")
plt.title("Target distribution in train")
plt.ylabel("Density")

# Define the combined train/test dataset for Proprocessing
combined = train.drop("SalePrice", axis=1).append(test)

# Check the percentage of missing data
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
