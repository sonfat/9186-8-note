# Exercise 8: Preparing data for support vector classifier

# clear environment prior to running this code

# import data
import pandas as pd
file_path = '../data_science/Data-Science-with-Python-master/Chapter03/weather.csv'
df = pd.read_csv(file_path)

# dummy code 'Summary'
import pandas as pd
df_dummies = pd.get_dummies(df, drop_first=True)

# shuffle df_dummies
from sklearn.utils import shuffle
df_shuffled = shuffle(df_dummies, random_state=42)

# split df_shuffled into X and y
DV = 'Rain' # Save the DV as DV
X = df_shuffled.drop(DV, axis=1) # get features (X)
y = df_shuffled[DV] # get DV (y)

# split X and y into testing and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# scale X_train and X_test
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # instantiate StandardScaler model
X_train_scaled = scaler.fit_transform(X_train) # transform X_train to z-scores
X_test_scaled = scaler.transform(X_test) # transform X_test to z-scores


# Exercise 9: Tuning support vector classifier using grid search

# continuing from exercise 8:

# instantiate grid
import numpy as np
grid = {'C': np.linspace(1, 10, 10),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

# instantiate GridSearchCV model
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
model = GridSearchCV(SVC(gamma='auto'), grid, scoring='f1', cv=5)


# fit the gridsearch model
model.fit(X_train_scaled, y_train)

# print the best parameters
best_parameters = model.best_params_
print(best_parameters)
# print(model)

# Activity 3: Generating predictions and evaluating performance of grid search SVC model

# continuing from Exercise 9:

# generate predicted classes
predicted_class = model.predict(X_test_scaled)

# evaluate performance with confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np
cm = pd.DataFrame(confusion_matrix(y_test, predicted_class))
cm['Total'] = np.sum(cm, axis=1)
cm = cm.append(np.sum(cm, axis=0), ignore_index=True)
cm.columns = ['Predicted No', 'Predicted Yes', 'Total']
cm = cm.set_index([['Actual No', 'Actual Yes', 'Total']])
print(cm)

# generate a classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predicted_class))