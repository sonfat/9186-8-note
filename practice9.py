import four4
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

file_path = '../../../Data-Science-with-Python-master/Chapter01/Data/Wholesale customers data.csv'
original_df = pd.read_csv(file_path, header=0)
# print(four4.get_4(original_df))
null_ = original_df.isna().any()
std_csale = StandardScaler()
minmax_scale = MinMaxScaler()

std_matrix = std_csale.fit_transform(original_df)
std_df = pd.DataFrame(std_matrix, columns=original_df.columns)

minmax_matrix = minmax_scale.fit_transform(original_df)
minmax_df = pd.DataFrame(minmax_matrix, columns=original_df.columns)
print(minmax_df.head(10))

