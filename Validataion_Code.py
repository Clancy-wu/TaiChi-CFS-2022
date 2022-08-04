#!/bin/python3
#author Clancy Wu
# Skitleran version 1.0.1

########### import #########################
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

########### Validation #########################
final_model = load('final_model.joblib')

mydata_train = pd.read_csv('x_train.csv')
mydata_test = pd.read_csv('x_test.csv')

x_train = mydata_train.drop('label',axis=1)
y_train = mydata_train['label'].values
x_test = mydata_test.drop('label',axis=1)
y_test = mydata_test['label'].values

np.mean(cross_val_score(final_model, x_train,y_train, cv=5, n_jobs=-1 )) # cross validation=0.95

step = 10
scores = [ ]
for i in range(step):
    score = np.mean(cross_val_score(final_model, x_train, y_train, cv=5, n_jobs=-1 ))
    scores.append(score)
    i  += 1
print('the min was : %s and the max was : %s' %(np.min(scores), np.max(scores)))
print('the sd was : %s ' %(np.std(scores)))
print('The 10 times running mean scores was: %s  ' %(np.mean(scores)))

step = 10
scores = [ ]
for i in range(step):
    final_model.fit(x_train,y_train)
    score = final_model.score(x_test,y_test)
    scores.append(score)
    i += 1
print('the min was : %s and the max was : %s' %(np.min(scores), np.max(scores)))
print('the sd was : %s ' %(np.std(scores)))
print('The 10 times running mean scores was: %s  ' %(np.mean(scores)))

########################################
### Permutation Test in test data
########################################
from sklearn.model_selection import permutation_test_score

final_model.fit(x_train, y_train)
score_df, perm_scores_df, pvalue_df = permutation_test_score(final_model, x_test, y_test, scoring="accuracy", cv=5, n_permutations=5000)
print('the P value of final model is : ', pvalue_df)

from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(x_train, y_train)
score_dummy, perm_scores_dummy, pvalue_dummy = permutation_test_score(dummy_clf, x_test, y_test, scoring="accuracy", cv=5, n_permutations=5000)
print('the P value of dummy_clf is : ', pvalue_dummy)

####################### End #################


