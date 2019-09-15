import pandas as pd
# Read file csv/excel
df_1 = pd.read_csv(<file.csv>, sep=';', dtype=<dictionary of columns>, decimal='.')
df_2 = pd.read_excel(<file.xls>, sheetname=<int or list of int>, dtype=<dictionary of columns>)

# Create a DataFrame
df_3 = pd.DataFrame(np.zeros(n,m))

# DataFrame basics
# Rename column: 
## inplace: Whether to return a new DataFrame. If True then value of copy is ignored.
df = df.rename(index=<dict-like or func> , columns={'col1':'new_col1','col2':'new_col2'...}, inplace=True)

# Slicing Ranges (can be used to assign values)
df[:5]          # first 5 rows
df[::2]         # select one row out of each 2 rows
df[::-1]        # inverse the row order
df[3:]          # rows starting from row 3
df[2:'A':'C']   # rows starting from row 2, column A and C

# Attribute Access and Assignment
df.iloc[n][m] = 1                     # integer-location based indexing for selection by position.
## create a new column
df['A'] = list(range(len(df.indx))) 
## modify column A if A already exists
df.A = list(range(len(df.indx))) 

## selection by callable
## select all the rows such that df.A>0
df = df.loc[lambda df: df['A']>0, :]  # Access a group of rows and columns by label(s) or a boolean array.
## select only the columns A and B
df = df.loc[:, lambda df: ['A','B']]
## for series
df = df.loc[lambda s: s>0]

## Interchanging column values without changing headers
df[['A','B']] = df[['B','A']]
## .loc and .iloc will not modify dataframe, correct way is to use raw values
df.loc[:,['B','A']] = df[['A','B']].values 
## Remove a column
df = df.drop(['id',...], axis=1) # axis=0 means index

