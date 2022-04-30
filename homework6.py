import pandas as pd
import numpy as np

# best_params = {'C': 4.0, 'penalty': 'l1', 'solver': 'liblinear'}
file_path = '../data_science/Data-Science-with-Python-master/Chapter03/weather.csv'
orinal_df = pd.read_csv(file_path)

# 虚拟编码
dummie_original_df = pd.get_dummies(orinal_df, drop_first=True)

# 打乱顺序
from  sklearn.utils import shuffle
shuffle_df = shuffle(dummie_original_df, random_state=42)

# 划分数据集
from sklearn.model_selection import train_test_split
DV = 'Rain'
X = shuffle_df.drop(DV, axis=1)
y = shuffle_df[DV]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 训练
from sklearn.linear_model import LogisticRegression
logi_model = LogisticRegression(C=4, penalty='l1', solver='liblinear')
logi_model.fit(X_train, y_train)

# 创建系数帧
intercept = logi_model.intercept_[0]
coeff = logi_model.coef_[0]
coeff_df = pd.DataFrame({'Feature': X_train.columns,
                         'coefficient': coeff})
# print(coeff_df)

# 预测
y_predict_proba = logi_model.predict_proba(X_test) # 预测概率对数
y_predict = logi_model.predict(X_test)

# 计算混淆矩阵
from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_test, y_predict, labels=[0,1])

confuse_df = pd.DataFrame(c_matrix)
confuse_df['Total'] = np.sum(confuse_df, axis=1)
confuse_df = confuse_df.append(np.sum(confuse_df, axis=0), ignore_index=True)
confuse_df.columns = ['predict_0', 'predict_1', 'Total']
confuse_df = confuse_df.set_index([['True_0', 'True_1', 'Total']])
print(confuse_df)
print('=' * 50)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, labels=[0,1]))
