import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
file_path = 'Data-Science-with-Python-master/Chapter01/Data/Banking_Marketing.csv'

df = pd.read_csv(file_path, header=0)
tmp_df = pd.read_csv(file_path, header=0)
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

onehot_encoder = OneHotEncoder(sparse=False)
# c = onehot_encoder.get_feature_names(df_catergory_columns)
oh_encoded = onehot_encoder.fit_transform(df[df_catergory_columns])
c = onehot_encoder.get_feature_names(df_catergory_columns)
oh_df = pd.DataFrame(oh_encoded, columns=c)

oh_df_getdummies = pd.get_dummies(tmp_df[df_catergory_columns], prefix=df_catergory_columns)
ont_hot_encode_data = pd.concat(oh_df_getdummies, df)



