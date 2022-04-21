import four4
import pandas as pd
import numpy as np

file_path = '../data_science/Data-Science-with-Python-master/Chapter01/Data/Banking_Marketing.csv'

# 1
original_df = pd.read_csv(file_path)
# print(original_df.shape)

# 2 done

# 3
# print(original_df.isna().sum())  # age:2, contact:6, duration:7

# 4
full_original_df = original_df.dropna(axis=0)
# print(full_original_df.shape)

# 5
# print(full_original_df['education'].value_counts())

# 6
full_original_df.education.replace({'basic.9y':'basic', 'basic.4y':'basic', 'basic.6y':'basic'}, inplace=True)
print(full_original_df['education'].value_counts())

# 7
category_columns = full_original_df.select_dtypes(exclude=[np.number]).columns
category_matrix = full_original_df[category_columns]
category_df = pd.DataFrame(category_matrix, columns=category_columns)
print(category_df.shape)

from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(sparse=False)
oh_category_matrix = oh.fit_transform(category_matrix)
print(oh_category_matrix.shape)
# oh_category_df = pd.DataFrame(oh_category_matrix, columns=category_columns)
# print(oh_category_df)
# oh_category_df = pd.DataFrame(oh_category_matrix ,columns=category_columns)