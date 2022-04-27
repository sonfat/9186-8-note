import four4
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split




file_path = '../data_science/Data-Science-with-Python-master/Chapter03/weather.csv'
original_df = pd.read_csv(file_path)

# 分类变量编码
dummies_df = pd.get_dummies(original_df, drop_first=True)

shuffle_df = shuffle(dummies_df, random_state=42)

# 拆分变量，本例中预测温度Temperature_c
dv = 'Temperature_c'
y = shuffle_df[dv]
X = shuffle_df.drop(dv, axis=1)
# 拆分数据集，训练:测试=2:1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 拟合简单线性回归模型
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train[['Humidity']], y_train) # 只用单一列做自变量
# print(X_train[['Humidity']].shape, y_train.shape)
intercept = linear_model.intercept_ # 截距
coefficient = linear_model.coef_ # 系数，输出为数列


# 进行预测
y_prediction = linear_model.predict(X_test[['Humidity']])



# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# plt.scatter(y_test, y_prediction)
# plt.xlabel('Y Test(True Value)')
# plt.ylabel('Y Prediction (Prediction Value)')
#
# pearson_value = pearsonr(y_test, y_prediction)[0]
#
# plt.title('True value VS Prediction Value (Pearson value = %0.2f ) '% (pearson_value))
# print(pearson_value)
# plt.show()

# import seaborn as sbn
# from scipy.stats import shapiro
# shapiro_p_value = shapiro(y_test-y_prediction)
# sbn.distplot((y_test-y_prediction), bins=50)
# plt.xlabel('Residuals')
# plt.ylabel('Density')
# plt.show()
# # print("{0:0.3f}".format(shapiro_p_value))
# # sbn.distplot()

# 计算平均绝对误差、均方误差、均方根差和拟合度
from sklearn import metrics

metrics_df = pd.DataFrame({'Metric': ['MAE',
                                      'MSE',
                                      'RMSE',
                                      'R-Squared'],
                           'Value':[metrics.mean_absolute_error(y_test, y_prediction),
                                    metrics.mean_squared_error(y_test, y_prediction),
                                    np.sqrt(metrics.mean_squared_error(y_test, y_prediction)),
                                    metrics.explained_variance_score(y_test, y_prediction)]})
print(metrics_df)