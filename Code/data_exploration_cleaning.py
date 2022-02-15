import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

train_loc = 'data/train.csv'
test_loc = 'data/test.csv'

train = pd.read_csv(train_loc, sep='\t')
test = pd.read_csv(test_loc, sep='\t')

#***************************************OVERVIEW***************************************#
#In this part, I will do some basic work to have an ouverview of the raw data.
train.describe()
test.describe()
#Overview finished.

#*************************************EXPLORATION**************************************#
important_cols = ['Anon Student Id',\
    'Problem Name','Problem View', 'Step Name', 'Step Start Time',\
        'First Transaction Time', 'Correct Transaction Time','Step End Time',\
            'Step Duration (sec)', 'Correct Step Duration (sec)',\
                'Error Step Duration (sec)','Correct First Attempt', 'Incorrects',\
                     'Hints', 'Corrects', 'KC(Default)', 'Opportunity(Default)']
datatype = {}
for col in train.columns:
    datatype[col] = train[col].dtype

num_of_students = len(np.unique(train['Anon Student Id']))
num_of_steps = len(np.unique(train[['Problem Name','Step Name']]))

'''
csd1 = train['Correct Step Duration (sec)']
print(csd1.describe())
hist = plt.hist(np.array(csd1.dropna()),bins=100,density=True,log=False,range=(0,100))
plt.xlabel('Time to correct answer (sec)')
plt.ylabel('Fraction')
plt.show()
'''
#The distribution of time duration of correct answer.
'''
csd2 = train['Error Step Duration (sec)']
print(csd2.describe())
hist = plt.hist(np.array(csd2.dropna()),bins=200,density=True,log=False,range=(0,200))
plt.xlabel('Time to incorrect answer (sec)')
plt.ylabel('Fraction')
plt.show()
'''
#The distribution of time duration of incorrect answer.
'''
csd3 = train['Correct First Attempt']
print(csd3.describe())
hist = plt.hist(np.array(csd3.dropna()),bins=2,density=True,log=False,range=(0,2))
plt.xlabel('Correct First Attempt')
plt.ylabel('Fraction')
plt.show()
'''
#A basic recognization of ratio of Correct First Attempt.
'''
csd4 = train['Problem View']
print(csd4.describe())
hist = plt.hist(np.array(csd4.dropna()),bins=22,density=True,log=False,range=(0,22))
plt.xlabel('Correct First Attempt')
plt.ylabel('Fraction')
plt.show()
'''
#Most of problem are only viewd once. 

#***************************************CLEANING***************************************#
#NOTICE: Following work wll be done:
#Check whether missing data(NaN) exists.
#Check whether the special variable is unique.
#Check and drop duplicate data.
#Check whether the data and variable types are consistent and reasonable.

#Part1: 
check_nan_train = train.isna().sum()
check_nan_test = test.isna().sum()
#Conclusion: For important columns: ['Anon Student Id','Problem Hierarchy','Problem Name','Problem View',\
# 'Step Name','Correct First Attempt'], there is no NaN exists(except for NaN Correct First Attempt in test.csv for test).
#Some columns have NaN values but that makes sense. 
#For example, if the student correctly finished the step at first attempt, the Error Step Duration should be none.

#Part2:
check_unique_train_spe = train.duplicated(subset=['Anon Student Id','Problem Name','Step Name','Problem View','Step Duration (sec)']).any()
check_unique_test_spe = test.duplicated(subset=['Anon Student Id','Problem Name','Step Name','Problem View','Step Duration (sec)']).any()
#Conclusion: The primary key of the data should be the combination of ['Anon Student Id','Problem Name','Step Name',\
# 'Problem View','Step Duration (sec)']. And given such index, each row in train and index is unique.

#Part3:
check_unique_train = len(train.drop_duplicates()) == len(train)
check_unique_test = len(test.drop_duplicates()) == len(test)
#Conclusion: For train and test, each row is unique with no duplicates.

#Part4:
cols = train.columns
'''cols = ['Row', 'Anon Student Id', 'Problem Hierarchy',\
    'Problem Name','Problem View', 'Step Name', 'Step Start Time',\
        'First Transaction Time', 'Correct Transaction Time','Step End Time',\
            'Step Duration (sec)', 'Correct Step Duration (sec)',\
                'Error Step Duration (sec)','Correct First Attempt', 'Incorrects',\
                     'Hints', 'Corrects', 'KC(Default)', 'Opportunity(Default)']'''
col_with_numerical_type = []
for col in cols:
    if train[col].dtype != 'object':
        col_with_numerical_type.append(col)
col_with_numerical_type.remove('Row')
'''col_with_numerical_type = ['Problem View','Step Duration (sec)', \
    'Correct Step Duration (sec)','Error Step Duration (sec)',\
        'Correct First Attempt', 'Incorrects','Hints', 'Corrects']'''
for col in col_with_numerical_type:
    tempmin = train[col][0]
    tempmax = train[col][0]
    for index,row in train.iterrows():
        if row[col] > tempmax:
            if row[col] - tempmax > 500:
                print(col,row[col],tempmax,'error')
            tempmax = row[col]
        elif tempmin > row[col]:
            if tempmin - row[col] > 500:
                print(col,row[col],tempmin,'error')
            tempmin = row[col]
for col in col_with_numerical_type:
    tempmin = train[col][0]
    tempmax = train[col][0]
    for index,row in test.iterrows():
        if not math.isnan(row[col]):
            if row[col] > tempmax:
                if row[col] - tempmax > 500:
                    print(col,row[col],tempmax,'error')
                tempmax = row[col]
            elif tempmin > row[col]:
                if tempmin - row[col] > 500:
                    print(col,row[col],tempmin,'error')
                tempmin = row[col]
#Conclusion: In this part, I focus on numerical_type columns. I set up a naive abnormal detection for train.csv and test.csv.
#If the value is extremely higher than tempmax or lower than tempmin, then I will print('error').
#After my detection, there is no value that cannot pass this naive test.
#It makes sense as well. The data in this project should be some kind of consecutive. For example, 'Problem View' should increase each time by 1.