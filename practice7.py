import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
file_path = 'Data-Science-with-Python-master/Chapter01/Data/Banking_Marketing.csv'

df = pd.read_csv(file_path, header=0)
# age和duration需要填补
# age用平均数填充
# print(list(df.age.describe()))
# print(df['age'].isna().sum())
df = df.dropna()
df_catergory_columns = df.select_dtypes(exclude=[np.number]).columns
# print(df_catergory)

label_encoder = LabelEncoder()
for i in df_catergory_columns:
    df[i] = label_encoder.fit_transform(df[i])
print(df.head())
