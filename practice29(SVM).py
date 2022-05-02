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
std_X_test = std_scaler.transform(X_test)
# print(std_X_train)

# 训练模型，确定最优参数
'''
__init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
'''
from sklearn.svm import SVC # SVM模型
from sklearn.model_selection import GridSearchCV # 超参数类
grid = {'C': np.linspace(1, 10, 10),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
grid_model = GridSearchCV(SVC(gamma='auto'), param_grid=grid, scoring='f1', cv=5)
grid_model.fit(std_X_train, y_train)
best_param = grid_model.best_params_
print(best_param)

svc_model = SVC(gamma='auto', probability=False, C=1.0, kernel='linear')
svc_model.fit(X_train, y_train)
y_predict = grid_model.predict(std_X_test)


# 输出报告
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
####混淆矩阵
cf_matrix = confusion_matrix(y_test, y_predict, labels=[0,1])

cf_df = pd.DataFrame(cf_matrix)
cf_df['Total'] = np.sum(cf_df, axis=1)
cf_df.columns = ['predict_0', 'predict_1', 'Total']
cf_df = cf_df.append(np.sum(cf_df, axis=0), ignore_index=True)
cf_df.set_index([['True_0', 'True_1', 'Total']])
print(cf_df)

report = classification_report(y_test, y_predict)
print(report)