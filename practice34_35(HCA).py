import pandas as pd


file_path = "raw_data/Chapter04/glass.csv"
origin_df = pd.read_csv(file_path)

# 乱序
from sklearn.utils import shuffle
shuffle_df = shuffle(origin_df, random_state=42)

# 标准化
from sklearn.preprocessing import StandardScaler
stder = StandardScaler()
std_df = stder.fit_transform(shuffle_df)

# 层次聚类分析HAC模型
from scipy.cluster.hierarchy import linkage
modle = linkage(std_df, method='complete')

# 作图
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram #系统树图
plt.figure(figsize=(10, 5))
plt.title('dendrogram for glass data')
dendrogram(modle, leaf_rotation=90, leaf_font_size=7)
plt.show()

# 生成与shuffle后的数据帧的行相对应的标签数组
from scipy.cluster.hierarchy import fcluster
labels = fcluster(modle, t=9, criterion='distance')
shuffle_df['Predicted_Cluster'] = labels
print(shuffle_df)
