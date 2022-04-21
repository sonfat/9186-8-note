import four4
import pandas as pd
from sklearn.model_selection import train_test_split




file_path = '../data_science/Data-Science-with-Python-master/Chapter01/Data/USA_Housing.csv'

original_df = pd.read_csv(file_path)

X = original_df.drop(labels=['Price'], axis=1)
y = original_df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train)