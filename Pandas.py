import pandas as pd
# Read file csv/excel
df_1 = pd.read_csv(<file.csv>, sep=';', dtype=<dictionary of columns>, decimal='.')
df_2 = pd.read_excel(<file.xls>, sheetname=<int or list of int>, dtype=<dictionary of columns>)
