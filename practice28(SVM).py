import pandas as pd
import numpy as np

file_path = '../data_science/Data-Science-with-Python-master/Chapter03/weather.csv'
orinal_df = pd.read_csv(file_path)

# 虚拟编码
dummie_df = pd.get_dummies(orinal_df, drop_first=True)

# 乱序
from sklearn.utils import shuffle
shuffle_df = shuffle(dummie_df, random_state=42)

# 拆分数据
from sklearn.model_selection import train_test_split
dv = 'Rain'
X = shuffle_df.drop(dv, axis=1)
y = shuffle_df[dv]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 标准化数据(z-score),服从标准正态分布N~(0,1)
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
std_X_train = std_scaler.fit_transform(X_train)
std_X_test = std_scaler.fit_transform(X_test)

