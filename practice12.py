import four4
import pandas as pd




file_path = '../../../Data-Science-with-Python-master/Chapter01/Data/USA_Housing.csv'

original_df = pd.read_csv(file_path)

X = original_df.drop(labels=['Price'], axis=1)