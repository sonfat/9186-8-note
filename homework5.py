import numpy as np
import pandas as pd

file_path = '../data_science/Data-Science-with-Python-master/Chapter03/weather.csv'
orinal_df = pd.read_csv(file_path)

# 对Description  10000 non-null object 进行编码
dummie_original_df = pd.get_dummies(orinal_df, drop_first=True)

# 打乱顺序
from sklearn.utils import shuffle
shuffle_df = shuffle(dummie_original_df, random_state=42)

# 拆分数据集
y = shuffle_df['Temperature_c']
X = shuffle_df.drop('Temperature_c', axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



# 训练模型
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
# intercept = linear_model.intercept_ # 截距
# coeffient = linear_model.coef_ # 系数
#
# x_ = list(X.columns)
# b_= list(coeffient)
# tmp = []
# for i in range(len(x_)):
#     tmp.append('(%.2f*%s)' % (b_[i], x_[i]))
# print('Tmeperature_c = ' + '+'.join(tmp))

#1 预测
y_prediction = linear_model.predict(X_test)
print(y_prediction)

#2 绘制真实值-预测值图像和计算皮尔逊系数
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats


#3 绘制残差图
fig, axes = plt.subplots(1,2)
pccs = scipy.stats.pearsonr(y_prediction, y_test)[0]
axes[0].set_title('True value VS Prediction value (PCCS = %0.2f)'.format(pccs) )
axes[0].set_xlabel('True value')
axes[0].set_ylabel('Prediction value')
axes[0].scatter(y_test, y_prediction)

axes[1].set_xlabel('Prediction')
axes[1].set_ylabel('Density')
sns.distplot((y_test - y_prediction), bins=50)
p_value = scipy.stats.shapiro((y_test - y_prediction))[1]
axes[1].set_title('Histogram of Prediction (p-value=%0.3f)' % p_value)
# plt.show()

#4 计算各项值
from sklearn import metrics

metrics_df = pd.DataFrame({'Metric':['MAE(平均绝对误差)',
                                     'MSE(均方误差)',
                                     'RMSE(均方根误差)',
                                     'R-Squared(拟合度)'],
                           'Value:':[metrics.mean_absolute_error(y_test, y_prediction),
                                     metrics.mean_squared_error(y_test, y_prediction),
                                     np.sqrt(metrics.mean_squared_error(y_test, y_prediction)),
                                     metrics.explained_variance_score(y_test,y_prediction)]})
print(metrics_df)


metrics_df2 = pd.DataFrame({'Metric':['MAE(平均绝对误差)',
                                     'MSE(均方误差)',
                                     'RMSE(均方根误差)',
                                     'R-Squared(拟合度)'],
                           'Value:':[metrics.mean_absolute_error(y_prediction, y_test),
                                     metrics.mean_squared_error(y_prediction, y_test),
                                     np.sqrt(metrics.mean_squared_error(y_prediction, y_test)),
                                     metrics.explained_variance_score(y_prediction, y_test)]})

print('='*50)
print(metrics_df2)