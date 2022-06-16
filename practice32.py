import numpy as np
import pandas as pd

file_path = '../data_science/Data-Science-with-Python-master/Chapter03/weather.csv'
origin_df = pd.read_csv(file_path)


# 虚拟编码
dummies_df = pd.get_dummies(origin_df)

# 乱序
from sklearn.utils import shuffle
shuffle_df = shuffle(dummies_df, random_state=42)

# 拆分数据
from sklearn.model_selection import train_test_split
dv = 'Temperature_c'
X = shuffle_df.drop(dv, axis=1)
y = shuffle_df[dv]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 标准化
from sklearn.preprocessing import StandardScaler
stder = StandardScaler()
std_X_train = stder.fit_transform(X_train)
std_X_test = stder.transform(X_test)

# 用随机森林算法，通过网格化调优参数
# from sklearn.ensemble import RandomForestClassifier # 分类器
'''
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators='warn',
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)
'''
grid ={'criterion':['mse', 'mae'],
       'max_features':['auto', 'sqrt', 'log2', None],
       'min_impurity_decrease':np.linspace(0.0, 1.0, 10),
       'bootstrap':[True, False],
       'warm_start':[True, False]}
from sklearn.ensemble import RandomForestRegressor #回归器
from sklearn.model_selection import GridSearchCV
modle = GridSearchCV(RandomForestRegressor(), grid, scoring='explained_variance', cv=5)
modle.fit(std_X_train, y_train)
best_parameters = modle.best_params_
print(best_parameters)

# result:{'bootstrap': True, 'criterion': 'mse', 'max_features': 'sqrt', 'min_impurity_decrease': 0.0, 'warm_start': False}