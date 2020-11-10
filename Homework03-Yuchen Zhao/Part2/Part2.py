import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# load data
data = pd.read_csv('train.csv')
dataset = data.values

# split data into X and y
X = dataset[:5000, 0:94]
y = dataset[:5000, 94]

# encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(y)

# grid search
n_estimators = [100, 200, 300, 400, 500]
learning_rate = [0.0001, 0.001, 0.01, 0.1]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
result = []

# Part 1
# No parallel
start = time.time()
model = XGBClassifier(n_jobs=1)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=1, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)
elapsed = time.time() - start
print("Processor number:", 1, "Elasped time:", elapsed, "s")
result.append(elapsed)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# 2 CPU
start = time.time()
model = XGBClassifier(n_jobs=2)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=2, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)
elapsed = time.time() - start
print("Processor number:", 2, "Elasped time:", elapsed, "s")
result.append(elapsed)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# 4 CPU
start = time.time()
model = XGBClassifier(n_jobs=4)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=4, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)
elapsed = time.time() - start
print("Processor number:", 4, "Elasped time:", elapsed, "s")
result.append(elapsed)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# 8 CPU
start = time.time()
model = XGBClassifier(n_jobs=8)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=8, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)
elapsed = time.time() - start
print("Processor number:", 8, "Elasped time:", elapsed, "s")
result.append(elapsed)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# plot
n = [1, 2, 4, 8]
plt.plot(n, result)
plt.ylabel('elasped time(seconds)')
plt.xlabel('number of processors')
plt.title('XGBoost Training Speed vs Number of Processors')
plt.savefig('XGBoost_Training_Speed_vs_Number_of_Processors.png')

# Part 2
model = XGBClassifier(n_jobs=-1)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)

print("Best Score: %f using Best Parameter%s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
for mean in means:
    print("Mean: %f based on Best Parameter: %s" % (mean, grid_result.best_params_))

# plot
scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
for i, value in enumerate(learning_rate):
    plt.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
plt.legend()
plt.xlabel('number of trees')
plt.ylabel('Log Loss')
plt.savefig('n_estimators_vs_Learning_Rate.png')