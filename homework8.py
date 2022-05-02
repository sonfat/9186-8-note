import pandas as pd
import numpy as np

file_path = '../data_science/Data-Science-with-Python-master/Chapter03/weather.csv'
original_df = pd.read_csv(file_path)

# 虚拟编码
dummie_df = pd.get_dummies(original_df, drop_first=True)

# 乱序
from sklearn.utils import shuffle
shuffle_df = shuffle(dummie_df, random_state=42)

# 划分数据
dv = 'Rain'
X = shuffle_df.drop(dv, axis=1)
y = shuffle_df[dv]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 标准化
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
std_X_train = std_scaler.fit_transform(X_train)
std_X_test = std_scaler.transform(X_test)
print(std_X_train)

'''
criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort=False):
'''
grid = {'criterion': ['gini', 'entropy'],
        'min_weight_fraction_leaf': np.linspace(0.0, 0.5, 10),
        'min_impurity_decrease': np.linspace(0.0, 1.0, 10),
        'class_weight': ["balanced", None],
        'presort': [True, False]}

# 训练模型
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
grid_model = GridSearchCV(DecisionTreeClassifier(), param_grid=grid, scoring='f1', cv=10)

grid_model.fit(std_X_train, y_train)
print(grid_model.best_params_)
y_predict = grid_model.predict(std_X_test)

# 混淆矩阵
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_predict, labels=[0,1])
cf_df = pd.DataFrame(cf_matrix)
cf_df['Total'] = np.sum(cf_df, axis=1)
cf_df.columns = ['predict_0', 'predict_1', 'Total']
cf_df = cf_df.append(np.sum(cf_df, axis=0), ignore_index=True)
cf_df = cf_df.set_index([['ture_0', 'true_1', 'Total']])
print(cf_df)

# 打印报告
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
