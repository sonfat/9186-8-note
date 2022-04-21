import four4
import pandas as pd




file_path = '../../../Data-Science-with-Python-master/Chapter01/Data/Student_bucketing.csv'


# print(four4.get_4(file_path))
df = pd.read_csv(file_path)
df['bucket'] = pd.cut(df['marks'], 3, labels=['Poor',
                                              # 'Below_average',
                                              'Average',
                                              # 'Above_average',
                                              'Excellent'])
print(df.head())