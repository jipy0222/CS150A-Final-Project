import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyRegressor
import lightgbm
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV

train_loc = './data/train_pyspark.csv'    # origin path 'data/train_pyspark.csv'
test_loc = './data/test_pyspark.csv'      # origin path 'data/test_pyspark.csv'

train = pd.read_csv(train_loc, sep='\t')
test = pd.read_csv(test_loc, sep='\t')


#************************************DATA_SEPERATING***********************************#
#We need to generate train_x, train_y, validation_x, validation_y.
train_x = train.dropna()
train_y = np.array(train_x['Correct First Attempt']).astype(int)
train_x = train_x.drop(['Correct First Attempt'],axis = 1)
valid_x = test.dropna()
valid_y = np.array(valid_x['Correct First Attempt']).astype(int)
valid_x = valid_x.drop(['Correct First Attempt'],axis = 1)
#Data seperating finished.

# data exploration

#***********************************DECOMPOSITION_PCA**********************************#
#Because the redundancy of features, we try to use PCA to decompose features.
#NOTICE: We use PoV(Proportion of Variance) as 0.9 to decompose features.
# pca = PCA(n_components=12)
# pca.fit(np.array(train_x))
# train_x = pca.transform(np.array(train_x))
# valid_x = pca.transform(np.array(valid_x))
# Decomposition finished.
# We used to try PCA, but it is not as good as expected.
'''
You may need to run preprocess.py first!(If you have something wrong, call me)
What you need to do is try different models, train and compare their performance.(on validation_Set by RMSE)
First step: Choose several models.(Recommend: Basic Decision Tree, Random Forest, Logistic Regression, \
    Simple Neural Network, SVM, LightGBM and maybe nonparametric methods)
Second step: Compare their performance and choose several best of them.
Third step: Select hyperparameter for the models you choosed in second step and find the best model and corresponding parameters.
Fourth step: Make prediction on test_set and do output.
'''

#*************************************MODEL_TRAINING************************************#


def normalize(x):
    norm_num = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    return x / norm_num


# RMSE error function
def RMSE(x, y):
    return np.sqrt(np.mean(np.square(np.subtract(x, y))))


def export():
    test_x = test
    test_y = np.array(test_x['Correct First Attempt']).astype(float)
    test_x = test_x.drop(['Correct First Attempt'], axis=1)
    model = RandomForestRegressor(n_estimators=190, max_depth=15, max_leaf_nodes=900)
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)
    for index, val in enumerate(test_y):
        if np.isnan(val):
            test_y[index] = test_res[index]
    new_test = pd.read_csv('./test.csv', sep='\t')
    new_test['Correct First Attempt'] = test_y
    new_test.to_csv('output.csv', sep='\t', index=False)

# normalize

# train_x = normalize(train_x)
# valid_x = normalize(valid_x)

# we used to try normalize the input vector, but not as good as we expected

# data exploration
print(train_x.shape[:])
print(valid_x.shape[:])
print(train_x)

# MLPRegressor

model = MLPRegressor(hidden_layer_sizes=(100, 5, 100), activation='tanh', solver='adam')
model.fit(train_x, train_y)
valid_res = model.predict(valid_x)

print("MLPRegressor error is %f" % RMSE(valid_res, valid_y))

# decision tree

model = tree.DecisionTreeClassifier()
model.fit(train_x, train_y)
valid_res = model.predict(valid_x)

print("decision tree error is %f" % RMSE(valid_res, valid_y))

# decision tree optimize

# max_depth = range(5, 21)
# min_samples_leaf = range(1, 10, 2)
# tree_para = {'criterion':['gini','entropy'], 'splitter':['best', 'random'], 'max_depth':max_depth, 'min_samples_leaf':min_samples_leaf}
# model = GridSearchCV(tree.DecisionTreeClassifier(), tree_para, n_jobs=-1)
# model.fit(train_x, train_y)
# valid_res = model.predict(valid_x)
#
# print("optimze decision tree error is %f" % RMSE(valid_res, valid_y))
# print("best parameters", model.best_params_ )

# RandomForestRegressor

model = RandomForestRegressor()
model.fit(train_x, train_y)
valid_res = model.predict(valid_x)

print("RandomForest error is %f" % RMSE(valid_res, valid_y))

# RandomForestRegressor optimize

# n_estimators = range(10, 200, 10)
# random_forest_para = {'n_estimators':n_estimators}
# model = GridSearchCV(RandomForestRegressor(), random_forest_para, n_jobs=-1)
# model.fit(train_x, train_y)
# valid_res = model.predict(valid_x)
#
# print("optimze RandomForest error is %f" % RMSE(valid_res, valid_y))
# print("best parameters", model.best_params_ )

# Adaboost

model = AdaBoostRegressor()
model.fit(train_x, train_y)
valid_res = model.predict(valid_x)

print("Adaboost error is %f" % RMSE(valid_res, valid_y))

# XGBoost

model = XGBClassifier()
model.fit(train_x, train_y)
valid_res = model.predict(valid_x)

print("XGBoost error is %f" % RMSE(valid_res, valid_y))

# lightgbm

model = lightgbm.LGBMClassifier()
model.fit(train_x, train_y)
valid_res = model.predict(valid_x)

print("lightgbm error is %f" % RMSE(valid_res, valid_y))

# Gradient Decision Tree

model = GradientBoostingClassifier(n_estimators=200)
model.fit(train_x, train_y)
valid_res = model.predict(valid_x)

print("Gradient Decision Tree error is %f" % RMSE(valid_res, valid_y))

# Logistic Regression

model = LogisticRegression(penalty='l2')
model.fit(train_x, train_y)
valid_res = model.predict(valid_x)

print("Logistic Regression error is %f" % RMSE(valid_res, valid_y))

# DummyRegressor()

model = DummyRegressor()
model.fit(train_x, train_y)
valid_res = model.predict(valid_x)

print("Dummy Regression error is %f" % RMSE(valid_res, valid_y))

# KNN

model = neighbors.KNeighborsRegressor()
model.fit(train_x, train_y)
valid_res = model.predict(valid_x)

print("KNN error is %f" % RMSE(valid_res, valid_y))

# bagging choice

# lr = LogisticRegression(penalty='l2', max_iter=100)
# model = BaggingRegressor(base_estimator=lr, n_estimators=100, max_samples=1.0, bootstrap=True)
# model.fit(train_x, train_y)
# valid_res = model.predict(valid_x)

# logistic model hard to converge

# print("logistic bagging error is %f" % RMSE(valid_res, valid_y))

tree = tree.DecisionTreeClassifier()
model = BaggingRegressor(base_estimator=tree, n_estimators=100, max_samples=1.0, bootstrap=True)
model.fit(train_x, train_y)
valid_res = model.predict(valid_x)

print("tree bagging error is %f" % RMSE(valid_res, valid_y))

# tree bagging optimize

# n_estimators = [150, 160, 170, 180, 190, 200]
# max_samples = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# bootstrap = [0, 1]
# # 'max_samples':max_samples, 'max_features':max_samples
# tree_bagging_para = {'n_estimators':n_estimators}
# # model = GridSearchCV(BaggingRegressor(base_estimator=tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.9), tree_bagging_para, n_jobs=-1)
# model = BaggingRegressor(base_estimator=tree.DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_leaf=5, splitter='random'), n_estimators=250, max_samples=0.5, max_features=0.9)
# model.fit(train_x, train_y)
# valid_res = model.predict(valid_x)
#
# print("optimize decision tree error is %f" % RMSE(valid_res, valid_y))
# print("best parameters", model.best_params_)

knn = neighbors.KNeighborsRegressor()
model = BaggingRegressor(base_estimator=knn, n_estimators=100, max_samples=1.0, bootstrap=True)
model.fit(train_x, train_y)
valid_res = model.predict(valid_x)

print("knn bagging error is %f" % RMSE(valid_res, valid_y))

Dummy = DummyRegressor()
model = BaggingRegressor(base_estimator=Dummy, n_estimators=100, max_samples=1.0, bootstrap=True)
model.fit(train_x, train_y)
valid_res = model.predict(valid_x)

print("Dummy bagging error is %f" % RMSE(valid_res, valid_y))

# export
export()