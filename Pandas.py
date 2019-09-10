import pandas as pd
# Read file csv/excel
df_1 = pd.read_csv(<file.csv>, sep=';', dtype=<dictionary of columns>, decimal='.')
df_2 = pd.read_excel(<file.xls>, sheetname=<int or list of int>, dtype=<dictionary of columns>)

# Create a DataFrame
df_3 = pd.DataFrame(np.zeros(n,m))

# DataFrame basics
# Rename column: 
# inplace: Whether to return a new DataFrame. If True then value of copy is ignored.
df.rename(index=<dict-like or func> , columns={'col1':'new_col1','col2':'new_col2'...}, inplace=True)

# Assignment
df.iloc[n][m] = 1

# Remove a column
df = df.drop(['id',...], axis=1) # axis=0 means index
