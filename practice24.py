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
intercept = linear_model.intercept_ # 截距
coeffient = linear_model.coef_ # 系数

x_ = list(X.columns)
b_= list(coeffient)
tmp = []
for i in range(len(x_)):
    tmp.append('(%.2f*%s)' % (b_[i], x_[i]))
print('Tmeperature_c = ' + '+'.join(tmp))
