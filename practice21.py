import four4
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split




file_path = '../data_science/Data-Science-with-Python-master/Chapter03/weather.csv'
original_df = pd.read_csv(file_path)

# 分类变量编码
dummies_df = pd.get_dummies(original_df, drop_first=True)

shuffle_df = shuffle(dummies_df, random_state=42)

# 拆分变量，本例中预测温度Temperature_c
dv = 'Temperature_c'
y = shuffle_df
X = shuffle_df.drop(dv, axis=1)
# 拆分数据集，训练:测试=2:1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 拟合简单线性回归模型
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
# linear_model.fit(X_train['Humidity'], y_train) # 只用单一列做自变量
# intercept = linear_model.intercept_ # 截距
# coefficient = linear_model.coef_ # 系数，输出为数列
# print(intercept[0])
# print(coefficient[0])

a = X_train['Humidity'].shape
b = X_train[['Humidity']].shape
print(a)
print(b)