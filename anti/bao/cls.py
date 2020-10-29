#-*- coding: UTF-8 -*-
import warnings
import random
from drawpic import rocauc
from drawpic import crossrocauc
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
import lightgbm as lgb
import numpy as np
import sklearn
from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import roc_auc_score,accuracy_score
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from sklearn import datasets
import matplotlib.pyplot as plt
from evaluation import evaluate

def classify(filepath,methods=['lightgbm']):
    # filepath = [neg-test...,neg-train...,pos-test...,pos-train...]
    # method = 
    trainsdata,traintags,testsdata,testtags = dataprocessing(filepath)
    methodset = ["svm","randomforest","gradientboosting","lightgbm","xgboost","decisiontree","bayes","adaboost"]
    data = {}
    for method in methods:
        dic = eval(method)(trainsdata,traintags,testsdata,testtags)
        data[method] = dic
    return data
def dataprocessing(filepath):
    print ("Loading feature files")
    dataset1 = pd.read_csv(filepath[0],header=None,low_memory=False)
    dataset2 = pd.read_csv(filepath[1],header=None,low_memory=False)
    dataset3 = pd.read_csv(filepath[2],header=None,low_memory=False)
    dataset4 = pd.read_csv(filepath[3],header=None,low_memory=False)
    # dataset1 = pd.read_csv('neg-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset2 = pd.read_csv('neg-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset3 = pd.read_csv('pos-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset4 = pd.read_csv('pos-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset1=pd.DataFrame(dataset1,dtype=np.float)
    print ("Feature processing")
    dataset1 = dataset1.convert_objects(convert_numeric=True)
    dataset2 = dataset2.convert_objects(convert_numeric=True)
    dataset3 = dataset3.convert_objects(convert_numeric=True)
    dataset4 = dataset4.convert_objects(convert_numeric=True)
    dataset1.dropna(inplace = True)
    dataset2.dropna(inplace = True)
    dataset3.dropna(inplace = True)
    dataset4.dropna(inplace = True)

    trainsdata = pd.concat([dataset2, dataset4],axis=0)

    negtraintags = [0]*dataset2.shape[0]
    postraintags= [1]*dataset4.shape[0]
    traintags = negtraintags+postraintags
    testsdata = pd.concat([dataset1, dataset3])
    negtesttags = [0]*dataset1.shape[0]
    postesttags= [1]*dataset3.shape[0]
    testtags = negtesttags+postesttags
    # 打乱数据集
    # cc = list(zip(trainsdata.values, traintags))
    # random.shuffle(cc)
    # trainsdata, traintags = zip(*cc)
    # #print type(trainsdata)
    # #print trainsdata.shape
    # trainsdata, traintags =pd.DataFrame(list(trainsdata)),list(traintags)
    # print trainsdata
    # print traintags
    return trainsdata,traintags,testsdata,testtags
def dataprocessingCV(filepath):
    print ("Loading feature files")
    #dataset1 = pd.read_csv(filepath[0],header=None,low_memory=False)
    # neg-train
    dataset2 = pd.read_csv(filepath[0],header=None,low_memory=False)
    # pos-train
    #dataset3 = pd.read_csv(filepath[2],header=None,low_memory=False)
    dataset4 = pd.read_csv(filepath[1],header=None,low_memory=False)
    # dataset1 = pd.read_csv('neg-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset2 = pd.read_csv('neg-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset3 = pd.read_csv('pos-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset4 = pd.read_csv('pos-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset1=pd.DataFrame(dataset1,dtype=np.float)
    print ("Feature processing")
    #dataset1 = dataset1.convert_objects(convert_numeric=True)
    dataset2 = dataset2.convert_objects(convert_numeric=True)
    #dataset3 = dataset3.convert_objects(convert_numeric=True)
    dataset4 = dataset4.convert_objects(convert_numeric=True)
    #dataset1.dropna(inplace = True)
    dataset2.dropna(inplace = True)
    #dataset3.dropna(inplace = True)
    dataset4.dropna(inplace = True)

    trainsdata = pd.concat([dataset2, dataset4],axis=0)

    negtraintags = [0]*dataset2.shape[0]
    postraintags= [1]*dataset4.shape[0]
    traintags = negtraintags+postraintags
    #testsdata = pd.concat([dataset1, dataset3])
    #negtesttags = [0]*dataset1.shape[0]
    #postesttags= [1]*dataset3.shape[0]
    #testtags = negtesttags+postesttags
    # 打乱数据集
    # cc = list(zip(trainsdata.values, traintags))
    # random.shuffle(cc)
    # trainsdata, traintags = zip(*cc)
    # #print type(trainsdata)
    # #print trainsdata.shape
    # trainsdata, traintags =pd.DataFrame(list(trainsdata)),list(traintags)
    # print trainsdata
    # print traintags
    return trainsdata,traintags


def logisticregression(trainsdata,traintags,testsdata,testtags):
    from sklearn import linear_model
    print("logisticregression")
    model = linear_model.LogisticRegression()
    model.fit(trainsdata,traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)

def xgboost(trainsdata,traintags,testsdata,testtags,params):
    xgtrain = xgb.DMatrix(trainsdata, label=traintags)
    xgtest = xgb.DMatrix(testsdata)
#     params = {'booster': 'gbtree',
#               'objective': 'binary:logistic',
# #              'eval_metric': 'auc',
#               'gamma':0,
#               'random_state':50,
#               'learning rate':0.01,
#               'max_depth': 3,
#               'lambda': 10,
#               'subsample': 0.8,
#               'colsample_bytree': 0.8,
#               'min_child_weight': 1,
#               'eta': 0.025,
#               'seed': 27,
#               'nthread': -1,
#               'scale_pos_weight' :1,
#               'silent': 0}
#     watchlist = [(xgtrain, 'train')]
    #bst = xgb.train(params, xgtrain, num_boost_round=110, evals=watchlist)
    bst = xgb.train(params, xgtrain)
    # xgb = XGBClassifier()
    # xgb.fit(trainsdata,traintags)
    # 输出概率
    ypred = bst.predict(xgtest)
    # y_pred = xgb.predict(testsdata)
    # y_score = xgb.predict_proba(trainsdata)
    # print y_pred
    # print y_score
    y_score = ypred
    y_pred = (ypred >= 0.5) * 1
    return evaluate(testtags, y_pred,np.array(y_score))
    # 设置阈值, 输出一些评价指标，选择概率大于0.5的为1，其他为0类
    #y_pred = (ypred >= 0.5) * 1
def bayes(trainsdata, traintags,testsdata,testtags):
    print("GaussianNB")
    by = GaussianNB()
    print("Training")
    by.fit(trainsdata,traintags)
    y_pred = by.predict(testsdata)
    y_score = by.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)

def decisiontree(trainsdata,traintags,testsdata,testtags):
    print("decisiontree")
    model = DecisionTreeClassifier()
    print("print Training ")
    model.fit(trainsdata,traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)
def svm(trainsdata,traintags,testsdata,testtags):
    print ("Svm Classifier")
    from sklearn import svm
    model = svm.SVC(probability=True)
    print("Svm training")
    model.fit(trainsdata, traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)
def randomforest(trainsdata,traintags,testsdata,testtags):
    from sklearn.ensemble import RandomForestClassifier
    print ("RandomForestClassifier")
    rf0 = RandomForestClassifier(oob_score=True)
    print ("Training")
    rf0.fit(trainsdata,traintags)
    y_pred = rf0.predict(testsdata)
    y_score = rf0.predict_proba(testsdata)

    return evaluate(testtags, y_pred,y_score)
def adaboost(trainsdata,traintags,testsdata,testtags):
    from sklearn.ensemble import AdaBoostClassifier
    print("AdaBoostClassifier")
    clf = AdaBoostClassifier(n_estimators=100,learning_rate=0.1)
    print("Training")
    clf.fit(trainsdata,traintags)
    y_pred = clf.predict( testsdata)
    y_score = clf.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)
def gradientboosting (trainsdata,traintags,testsdata,testtags):
    from sklearn.ensemble import GradientBoostingClassifier
     #迭代100次 ,学习率为0.1
    print("GradientBoostingClassifier")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    print("Training")
    clf.fit(trainsdata,traintags)
    y_pred = clf.predict( testsdata)
    y_score = clf.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)
def lightgbm(traindata,traintags,testdata,testtags,params):
    train_data=lgb.Dataset(traindata,label=traintags)
    # validation_data=lgb.Dataset(testdata,label=testtags)
    #
    clf=lgb.train(params,train_data)
    from sklearn.metrics import roc_auc_score,accuracy_score
    # clf = LGBMClassifier()
    # clf.set_params(params)
    # clf.fit(traindata,traintags)
    y_pred=clf.predict(testdata,num_iteration=clf.best_iteration)
    y_score = y_pred
    print y_score
    for i in range(len(y_pred)):
        if y_pred[i]>0.5:y_pred[i]=1
        else:y_pred[i]=0

    return evaluate(testtags, y_pred,np.array(y_score))
#filepath=['featurefiles/neg-test.a.3.1.1.fasta.csv','featurefiles/neg-train.a.3.1.1.fasta.csv','featurefiles/pos-test.a.3.1.1.fasta.csv','featurefiles/pos-train.a.3.1.1.fasta.csv',]
#methodset = ["svm","randomforest","gradientboosting","lightgbm","xgboost","decisiontree","bayes","adaboost"]
#filepath=['featurefiles/neg-test.a.3.1.1.fasta.csv','featurefiles/neg-train.a.3.1.1.fasta.csv','featurefiles/pos-test.a.3.1.1.fasta.csv','featurefiles/pos-train.a.3.1.1.fasta.csv',]
filepath=['featurefiles/neg-test.txt.csv','featurefiles/neg-train.txt.csv','featurefiles/pos-test.txt.csv','featurefiles/pos-train.txt.csv',]
#
# #data = classify(filepath,methods = ["svm","randomforest","gradientboosting","lightgbm","xgboost","decisiontree","bayes","adaboost"]
# method = ["randomforest","gradientboosting","xgboost"]
# data = classify(filepath,methods = method)
# rocauc(data)
#KF = KFold(n_splits = 10,shuffle=True,random_state=10)
#data为数据集,利用KF.split划分训练集和测试集
# traindata, traintags, testdata, testtags = dataprocessing(filepath)


#
# for train_index,test_index in KF.split(traindata):
#     #建立模型，并对训练集进行测试，求出预测得分
#     #划分训练集和测试集
#     data = {}
#     #print test_index
#     X_train,X_test = np.array(traindata)[train_index],np.array(traindata)[test_index]
#     Y_train,Y_test = np.array(traintags)[train_index],np.array(traintags)[test_index]
#     #建立模型(模型已经定义)
#     # model = build_model()
#     # #编译模型
#     # model.compile(optimizer = 'sgd',loss = 'categorical_crossentropy',metrics = ['acc'])
#     # #训练模型
#     # model.fit(X_train,Y_train,batch_size = 2,validation_data = (X_test,Y_test),epochs = 150)
#     #利用model.predict获取测试集的预测值
#     dic = xgboost(X_train,Y_train,X_test,Y_test)
#     data['xgboost'] = dic
#     rocauc(data)
#     #计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
#     # fpr,tpr,thresholds=roc_curve(Y_test[:,1],y_pred[:,1])
#     # #interp:插值 把结果添加到tprs列表中
#     # tprs.append(interp(mean_fpr,fpr,tpr))
#     # tprs[-1][0]=0.0
#     # #计算auc
#     # roc_auc=auc(fpr,tpr)
#     # aucs.append(roc_auc)
#     # #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
#     # plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i,roc_auc))
#     # i +=1
# xgb = XGBClassifier(
#     learning_rate=0.1,
#     n_estimators=1,
#     max_depth=5,
#     # reg_alpha = 1e-5,
#     min_child_weight=1,
#     gamma=0,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective='binary:logistic',
#     nthread=-1,
#     scale_pos_weight=1,
#     seed=27)
# #print(sklearn.metrics.SCORERS.keys())
# #sco = cross_val_score(xgb,traindata,traintags,scoring='recall',cv=KF,n_jobs = -1)
# scos =cross_validate(xgb,traindata,traintags,scoring=['recall','precision'],cv=KF,n_jobs = -1)
# #testscore = scos['test_score']
# print scos