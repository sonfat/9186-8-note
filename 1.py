# Activity 4: Prepare data for decision tree classifier

# clear environment prior to running this code

# import data
import pandas as pd
df = pd.read_csv('../data_science/Data-Science-with-Python-master/Chapter03/weather.csv')

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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Exercise 10: Tuning decision tree classifier using grid search in pipeline

# continuing from Activity 4:

# Specify the hyperparameter space
import numpy as np
grid = {'criterion': ['gini', 'entropy'],
        'min_weight_fraction_leaf': np.linspace(0.0, 0.5, 10),
        'min_impurity_decrease': np.linspace(0.0, 1.0, 10),
        'class_weight': [None, 'balanced'],
        'presort': [True, False]}

# Instantiate the GridSearchCV model
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
model = GridSearchCV(DecisionTreeClassifier(), grid, scoring='f1', cv=5)

# Fit to the training set
model.fit(X_train_scaled, y_train)

# Print the tuned parameters
best_parameters = model.best_params_
print(best_parameters)

# Exercise 10: Tuning decision tree classifier using grid search in pipeline

# continuing from Activity 4:

# Specify the hyperparameter space
import numpy as np
grid = {'criterion': ['gini', 'entropy'],
        'min_weight_fraction_leaf': np.linspace(0.0, 0.5, 10),
        'min_impurity_decrease': np.linspace(0.0, 1.0, 10),
        'class_weight': [None, 'balanced'],
        'presort': [True, False]}

# Instantiate the GridSearchCV model
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
model = GridSearchCV(DecisionTreeClassifier(), grid, scoring='f1', cv=5)

# Fit to the training set
model.fit(X_train_scaled, y_train)

# Print the tuned parameters
best_parameters = model.best_params_
print(best_parameters)

# Exercise 11: Programmatically extracting tuned hyperparameters from decision tree classifier grid search model

# continuing from Exercise 10:

# access the 'Tree__criterion' value
print(best_parameters['criterion'])

# instantiate model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(class_weight=best_parameters['class_weight'],
                               criterion=best_parameters['criterion'],
                               min_impurity_decrease=best_parameters['min_impurity_decrease'],
                               min_weight_fraction_leaf=best_parameters['min_weight_fraction_leaf'],
                               presort=best_parameters['presort'])

# scale X_train and fit model
model.fit(X_train_scaled, y_train)

# extract feature_importances attribute
print(model.feature_importances_)

# plot feature importance in descending order
import pandas as pd
import matplotlib.pyplot as plt
df_imp = pd.DataFrame({'Importance': list(model.feature_importances_)}, index=X.columns)
# sort dataframe
df_imp_sorted = df_imp.sort_values(by=('Importance'), ascending=True)
# plot these
df_imp_sorted.plot.barh(figsize=(5,5))
plt.title('Relative Feature Importance')
plt.xlabel('Relative Importance')
plt.ylabel('Variable')
plt.legend(loc=4)
plt.show()






# Activity 5: Generating predictions and evaluating performance of decision tree classifier model

# continuing from Exercise 11:

# generate predicted probabilities of rain
predicted_prob = model.predict_proba(X_test_scaled)[:,1]

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


