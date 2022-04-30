import pandas as pd

file_path = '../data_science/Data-Science-with-Python-master/Chapter03/weather.csv'
orinal_df = pd.read_csv(file_path)

dummie_orinal_df = pd.get_dummies(orinal_df, drop_first=True)

from sklearn.utils import shuffle
shuffle_df = shuffle(dummie_orinal_df, random_state=42)

from sklearn.model_selection import train_test_split
DV = 'Rain'
X = shuffle_df.drop(DV, axis=1)
y = shuffle_df[DV]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
intercept = logistic_model.intercept_[0]
coeffient_list = logistic_model.coef_[0]
cof_df = pd.DataFrame({'Feature':X.columns,
                       'cof_':coeffient_list})
# print(cof_df)

# 预测概率
y_predict_prob = logistic_model.predict_proba(X_test)[:,1]

# 预测类别
y_predict = logistic_model.predict(X_test)

# 使用混淆矩阵评估
from sklearn.metrics import confusion_matrix
import numpy as np
c_matrix = confusion_matrix(y_test, y_predict, labels=[0,1]) # labels参数用于控制输出的顺序
cm_df = pd.DataFrame(c_matrix)
cm_df['Total'] = np.sum(cm_df, axis=1)
cm_df = cm_df.append(np.sum(cm_df, axis=0), ignore_index=True)
cm_df.columns = ['Predict_0', 'Predict_1', 'Total']
cm_df = cm_df.set_index([['Actual_0', 'Actual_1', 'Total']])
# print(cm_df)

from sklearn.metrics import classification_report
# print(classification_report(y_test, y_predict))

# 使用超参数
print(logistic_model)
grid = {'penalty': ['l1', 'l2'],
        'C': np.linspace(1, 10, 10),
        'solver': ['warn', 'liblinear']}
# 寻找F-1分数最高的参数
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
model = GridSearchCV(LogisticRegression(solver='liblinear'), grid, scoring='f1', cv=5)
model.fit(X_train, y_train)
best_value = model.best_params_
print(best_value)