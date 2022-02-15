from pyspark import SQLContext, SparkContext
import pyspark.sql.functions as F
from pyspark.sql.types import *
import numpy as np
import pandas

import findspark
findspark.init()

sparkc = SparkContext('local')
sqlc = SQLContext(sparkc)

train_loc = 'data/train.csv'
train = sqlc.read.csv(train_loc, sep='\t', header=True)
test_loc = 'data/test.csv'
test = sqlc.read.csv(test_loc, sep='\t', header=True)


#****************************************REMOVE****************************************#
#Because several columns are not provided by test, they should not be features to be considered, so we simply remove them.
#NOTICE: 'Correct First Attempt' should be train_label.
columns_to_drop = ['Row','Step Start time','First Transaction Time','Correct Transaction Time','Step End Time','Step Duration (sec)',\
    'Correct Step Duration (sec)','Error Step Duration (sec)','Incorrects','Hints','Corrects']

train = train.drop(*columns_to_drop)
test = test.drop(*columns_to_drop)
#Remove useless columns finished.


#***************************************SEPERATE***************************************#
#Divide column 'Problem Hierarchy' into two columns: 'Problem Unit' and 'Problem Section'.
#NOTICE: Using udf functions & We should drop 'Problem Hierarchy' in the end.
def sep_hier_unit(str): 
    return str.split(',')[0]
    
def sep_hier_section(str):
    return str.split(',')[1]

udfsep_hier_unit = F.udf(sep_hier_unit, StringType())
udfsep_hier_section = F.udf(sep_hier_section, StringType())

train = train.withColumn('Problem Unit', udfsep_hier_unit('Problem Hierarchy'))
train = train.withColumn('Problem Section', udfsep_hier_section('Problem Hierarchy'))
test = test.withColumn('Problem Unit', udfsep_hier_unit('Problem Hierarchy'))
test = test.withColumn('Problem Section', udfsep_hier_section('Problem Hierarchy'))

train = train.drop('Problem Hierarchy')
test = test.drop('Problem Hierarchy')
#Divide column 'Problem Hierarchy' finished.


#****************************************ENCODE****************************************#
#Because the existence of categorical feature, We need to enocde them.
#NOTICE: We should first union train and test together, otherwise we may generate different encoding for train and test.
def encode(column):
    global train,test
    ref_dict = {}
    ori_temp = train.union(test).select(column).distinct().collect()
    temp = []
    for item in ori_temp:
        temp.append(item[column])
    index = 1
    for item in temp:
        ref_dict[item] = index
        index += 1
    
    def innerencode(str):
        return ref_dict[str]
    
    udfinnerencode = F.udf(innerencode,IntegerType())
    
    train = train.withColumn('New '+column, udfinnerencode(column))
    train = train.drop(column)
    train = train.withColumnRenamed('New '+column, column)
    test = test.withColumn('New '+column, udfinnerencode(column))
    test = test.drop(column)
    test = test.withColumnRenamed('New '+column, column)

column_to_encode = ['Anon Student Id','Problem Name','Problem Unit','Problem Section','Step Name']
for time in range(len(column_to_encode)):
    encode(column_to_encode[time])
#Categorical column encoding finished.


#***************************************CALCULATE**************************************#
#Change KC and Opportunity to be easier to understand.
#NOTICE: I will add new features: KC_count and Opportunity_average.
def KC_count(str):
    if not str:
        return 0
    else:
        return str.count('~~')+1

def Opp_avg(str):
    if not str:
        return 0.0
    else:
        sum = 0
        count = 0
        strlist = str.split('~~')
        for item in strlist:
            sum += eval(item)
            count += 1
        return float(sum/count)

udfKC_count = F.udf(KC_count, IntegerType())
udfOpp_avg = F.udf(Opp_avg, FloatType())

train = train.withColumn('KC Count', udfKC_count('KC(Default)'))
test = test.withColumn('KC Count', udfKC_count('KC(Default)'))
train = train.withColumn('Opportunity Average', udfOpp_avg('Opportunity(Default)'))
train = train.drop('Opportunity(Default)')
test = test.withColumn('Opportunity Average', udfOpp_avg('Opportunity(Default)'))
test = test.drop('Opportunity(Default)')
#Calculate column finished.


#*************************************NEW_FEATURES*************************************#
#We will try to add several new features as below:
#Personal Correct First Attempt Count(Personal CFAC): which counts the num of correct first attempts by given student_id.
#Personal Correct First Attempt Rate(Personal CFAR): which measures the rate of correct first attempts by given student_id.
#Problem Correct First Attempt Rate(Problem CFAR): which measures the rate of correct first attempts by given problem_name.
#Unit Correct First Attempt Rate(Unit CFAR):  which measures the rate of correct first attempts by given problem_unit.
#Section Correct First Attempt Rate(Section CFAR):  which measures the rate of correct first attempts by given problem_section.
#Step Correct First Attempt Rate(Step CFAR): which measures the rate of correct first attempts by given step_name.
#KC Correct First Attempt Rate(KC CFAR): which measures the rate of correct first attempts by given KC.
#NOTICE: For generalization, we will consider step_name instead of (problem_name,step_name) when figuring out Step CFAR.
#NOTICE: For simplicity, we will consider original KC instead of the minimun KC CFAR of separated KC elements when figuring out KC CFAR.

benchmark = train.filter(train['Correct First Attempt'] == '1')

#Personal Correct First Attempt Count(Personal CFAC)
#NOTICE: For those names which don't show in train, use the average to estimate it. 
def get_Personal_Count():
    global train,test
    PC_dict = {}
    ori_temp = benchmark.groupBy('Anon Student Id').count().collect()
    ori_temp2 = train.groupBy('Anon Student Id').count().collect()
    for item in ori_temp:
        PC_dict[item['Anon Student Id']] = item['count']
    for item in ori_temp2:
        if item['Anon Student Id'] not in PC_dict:
            PC_dict[item['Anon Student Id']] = 0
    sum = 0
    for key in PC_dict.keys():
        sum += PC_dict[key]
    mean = float(sum/len(ori_temp2))
    
    def get_Id_Count(id):
        if id in PC_dict.keys():
            return float(PC_dict[id])
        else:
            return mean
        
    udfget_Id_Count = F.udf(get_Id_Count, FloatType())
    train = train.withColumn('Personal CFAC', udfget_Id_Count('Anon Student Id'))
    test = test.withColumn('Personal CFAC',udfget_Id_Count('Anon Student Id'))

get_Personal_Count()
#Personal CFAC finished.

#Personal Correct First Attempt Rate(Personal CFAR)
def get_Personal_Rate():
    global train,test
    PR_dict1 = {}
    PR_dict2 = {}
    PR_dict = {}
    ori_temp = benchmark.groupBy('Anon Student Id').count().collect()
    ori_temp2 = train.groupBy('Anon Student Id').count().collect()
    for item in ori_temp:
        PR_dict1[item['Anon Student Id']] = item['count']
    for item in ori_temp2:
        PR_dict2[item['Anon Student Id']] = item['count']
    for item in ori_temp:
        PR_dict[item['Anon Student Id']] = item['count']/PR_dict2[item['Anon Student Id']]
    for item in ori_temp2:
        if item['Anon Student Id'] not in PR_dict:
            PR_dict[item['Anon Student Id']] = 0
    sum = 0
    for key in PR_dict.keys():
        sum += PR_dict[key]
    mean = float(sum/len(ori_temp2))

    def get_Id_Rate(id):
        if id in PR_dict.keys():
            return float(PR_dict[id])
        else:
            return mean
        
    udfget_Id_Rate = F.udf(get_Id_Rate, FloatType())
    train = train.withColumn('Personal CFAR', udfget_Id_Rate('Anon Student Id'))
    test = test.withColumn('Personal CFAR',udfget_Id_Rate('Anon Student Id'))

get_Personal_Rate()
#Personal CFAR finished.

#Problem Correct First Attempt Rate(Problem CFAR)
def get_Problem_Rate():
    global train,test
    PR_dict1 = {}
    PR_dict2 = {}
    PR_dict = {}
    ori_temp = benchmark.groupBy('Problem Name').count().collect()
    ori_temp2 = train.groupBy('Problem Name').count().collect()
    for item in ori_temp:
        PR_dict1[item['Problem Name']] = item['count']
    for item in ori_temp2:
        PR_dict2[item['Problem Name']] = item['count']
    for item in ori_temp:
        PR_dict[item['Problem Name']] = item['count']/PR_dict2[item['Problem Name']]
    for item in ori_temp2:
        if item['Problem Name'] not in PR_dict:
            PR_dict[item['Problem Name']] = 0
    sum = 0
    for key in PR_dict.keys():
        sum += PR_dict[key]
    mean = float(sum/len(ori_temp2))

    def get_Name_Rate(id):
        if id in PR_dict.keys():
            return float(PR_dict[id])
        else:
            return mean
        
    udfget_Name_Rate = F.udf(get_Name_Rate, FloatType())
    train = train.withColumn('Problem CFAR', udfget_Name_Rate('Problem Name'))
    test = test.withColumn('Problem CFAR',udfget_Name_Rate('Problem Name'))

get_Problem_Rate()
#Problem CFAR finished.

#Unit Correct First Attempt Rate(Unit CFAR)
def get_Unit_Rate():
    global train,test
    PR_dict1 = {}
    PR_dict2 = {}
    PR_dict = {}
    ori_temp = benchmark.groupBy('Problem Unit').count().collect()
    ori_temp2 = train.groupBy('Problem Unit').count().collect()
    for item in ori_temp:
        PR_dict1[item['Problem Unit']] = item['count']
    for item in ori_temp2:
        PR_dict2[item['Problem Unit']] = item['count']
    for item in ori_temp:
        PR_dict[item['Problem Unit']] = item['count']/PR_dict2[item['Problem Unit']]
    for item in ori_temp2:
        if item['Problem Unit'] not in PR_dict:
            PR_dict[item['Problem Unit']] = 0
    sum = 0
    for key in PR_dict.keys():
        sum += PR_dict[key]
    mean = float(sum/len(ori_temp2))

    def get_Unitid_Rate(id):
        if id in PR_dict.keys():
            return float(PR_dict[id])
        else:
            return mean
        
    udfget_Unitid_Rate = F.udf(get_Unitid_Rate, FloatType())
    train = train.withColumn('Unit CFAR', udfget_Unitid_Rate('Problem Unit'))
    test = test.withColumn('Unit CFAR',udfget_Unitid_Rate('Problem Unit'))

get_Unit_Rate()
#Unit CFAR finished.

#Section Correct First Attempt Rate(Section CFAR)
def get_Section_Rate():
    global train,test
    PR_dict1 = {}
    PR_dict2 = {}
    PR_dict = {}
    ori_temp = benchmark.groupBy('Problem Section').count().collect()
    ori_temp2 = train.groupBy('Problem Section').count().collect()
    for item in ori_temp:
        PR_dict1[item['Problem Section']] = item['count']
    for item in ori_temp2:
        PR_dict2[item['Problem Section']] = item['count']
    for item in ori_temp:
        PR_dict[item['Problem Section']] = item['count']/PR_dict2[item['Problem Section']]
    for item in ori_temp2:
        if item['Problem Section'] not in PR_dict:
            PR_dict[item['Problem Section']] = 0
    sum = 0
    for key in PR_dict.keys():
        sum += PR_dict[key]
    mean = float(sum/len(ori_temp2))

    def get_Sectionid_Rate(id):
        if id in PR_dict.keys():
            return float(PR_dict[id])
        else:
            return mean
        
    udfget_Sectionid_Rate = F.udf(get_Sectionid_Rate, FloatType())
    train = train.withColumn('Section CFAR', udfget_Sectionid_Rate('Problem Section'))
    test = test.withColumn('Section CFAR',udfget_Sectionid_Rate('Problem Section'))

get_Section_Rate()
#Section CFAR finished

#Step Correct First Attempt Rate(Step CFAR):
def get_Step_Rate():
    global train,test
    PR_dict1 = {}
    PR_dict2 = {}
    PR_dict = {}
    ori_temp = benchmark.groupBy('Step Name').count().collect()
    ori_temp2 = train.groupBy('Step Name').count().collect()
    for item in ori_temp:
        PR_dict1[item['Step Name']] = item['count']
    for item in ori_temp2:
        PR_dict2[item['Step Name']] = item['count']
    for item in ori_temp:
        PR_dict[item['Step Name']] = item['count']/PR_dict2[item['Step Name']]
    for item in ori_temp2:
        if item['Step Name'] not in PR_dict:
            PR_dict[item['Step Name']] = 0
    sum = 0
    for key in PR_dict.keys():
        sum += PR_dict[key]
    mean = float(sum/len(ori_temp2))

    def get_Name_Rate(id):
        if id in PR_dict.keys():
            return float(PR_dict[id])
        else:
            return mean
        
    udfget_Name_Rate = F.udf(get_Name_Rate, FloatType())
    train = train.withColumn('Step CFAR', udfget_Name_Rate('Step Name'))
    test = test.withColumn('Step CFAR',udfget_Name_Rate('Step Name'))

get_Step_Rate()
#Step CFAR finished

#KC Correct First Attempt Rate(KC CFAR)
def get_KC_Rate():
    global train,test
    PR_dict1 = {}
    PR_dict2 = {}
    PR_dict = {}
    ori_temp = benchmark.groupBy('KC(Default)').count().collect()
    ori_temp2 = train.groupBy('KC(Default)').count().collect()
    for item in ori_temp:
        PR_dict1[item['KC(Default)']] = item['count']
    for item in ori_temp2:
        PR_dict2[item['KC(Default)']] = item['count']
    for item in ori_temp:
        PR_dict[item['KC(Default)']] = item['count']/PR_dict2[item['KC(Default)']]
    for item in ori_temp2:
        if item['KC(Default)'] not in PR_dict:
            PR_dict[item['KC(Default)']] = 0
    sum = 0
    for key in PR_dict.keys():
        sum += PR_dict[key]
    mean = float(sum/len(ori_temp2))

    def get_KCid_Rate(id):
        if id in PR_dict.keys():
            return float(PR_dict[id])
        else:
            return mean
        
    udfget_KCid_Rate = F.udf(get_KCid_Rate, FloatType())
    train = train.withColumn('KC CFAR', udfget_KCid_Rate('KC(Default)'))
    test = test.withColumn('KC CFAR',udfget_KCid_Rate('KC(Default)'))

get_KC_Rate()
train = train.drop('KC(Default)')
test = test.drop('KC(Default)')
#KC CFAR finished
#Add New_Features finished.

#**************************************MATERIALIZE*************************************#
train.toPandas().to_csv('data/train_pyspark.csv', sep='\t', header=True, index = False)
test.toPandas().to_csv('data/test_pyspark.csv', sep='\t', header=True, index = False)
#Materialization finished.