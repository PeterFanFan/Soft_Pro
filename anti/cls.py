#-*- coding: UTF-8 -*-
import warnings
from catboost import CatBoostClassifier
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.svm import SVC

from lightgbm.sklearn import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.neighbors import  KNeighborsClassifier
import lightgbm as lgb
import xgboost as xgb
from evaluation import evaluate
def mkd(x):
    if (os.path.exists(x)):
        pass
    else:
        os.makedirs(x)

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
# r
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

def knn(trainsdata,traintags,testsdata,testtags):
    model= KNeighborsClassifier()
    model.fit(trainsdata,traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)
def logisticregression(trainsdata,traintags,testsdata,testtags):
    from sklearn import linear_model
    # print("logisticregression")
    model = linear_model.LogisticRegression()
    model.fit(trainsdata,traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)
def trainmodel_multipleTags(traindata,traintags,testdata,testtags):

    model_Knn= KNeighborsClassifier()
    model_Knn.fit(traindata,traintags)
    train_lable_Knn = model_Knn.predict(traindata)
    y_pred_Knn = model_Knn.predict(testdata)

    model_G_NB = GaussianNB()
    model_G_NB.fit(traindata,traintags)
    train_lable_NB = model_G_NB.predict(traindata)
    y_pred_NB = model_G_NB.predict(testdata)


    model_SVM = SVC(probability=True)
    model_SVM.fit(traindata, traintags)
    train_lable_SVM = model_SVM.predict(traindata)
    y_pred_SVM = model_SVM.predict(testdata)

    model_DT = DecisionTreeClassifier()
    model_DT.fit(traindata,traintags)
    train_lable_DT = model_DT.predict(traindata)
    y_pred_DT = model_DT.predict(testdata)



    trainLableList=[train_lable_DT,train_lable_Knn,train_lable_NB,train_lable_SVM]
    y_predList=[y_pred_DT,y_pred_Knn,y_pred_NB,y_pred_SVM]

    return trainLableList,y_predList



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
    # print("GaussianNB")
    by = GaussianNB()
    # print("Training")
    by.fit(trainsdata,traintags)
    y_pred = by.predict(testsdata)
    y_score = by.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)

def decisiontree(trainsdata,traintags,testsdata,testtags):
    # print("decisiontree")
    model = DecisionTreeClassifier()
    # print("print Training ")
    model.fit(trainsdata,traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)
def svm_opt(trainsdata,traintags,testsdata,testtags):
    # print ("Svm Classifier")
    params = {'C': 9.928279022403954, 'gamma': 0.00014496253219114155}
    # params = {'C': 1, 'gamma': 0.00014496253219114155}
    #params = {'C': 10}
    print("hello")
    model = SVC(C=params['C'], gamma=params['gamma'], probability=True)

    model.fit(trainsdata, traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)
##after svm opt para
def svm_old_opt(testsdata,testtags,model):
    # print ("SvmOpt Classifier")
#     from sklearn import svm
#     model = svm.SVC(probability=True)
# #    print("Svm training")
#     model.fit(trainsdata, traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)

    from evaluation import evaluate
    return  evaluate(testtags, y_pred,y_score)
def sssvm(testsdata,testtags,model):
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)

    from evaluation import evaluate
    return  evaluate(testtags, y_pred,y_score)
import joblib
def svmOpt(trainsdata,traintags,testsdata,testtags):
    # print ("SvmOpt Classifier")
    from sklearn import svm
    model = svm.SVC(probability=True)
#    print("Svm training")
    model.fit(trainsdata, traintags)
    joblib.dump(model, 'model/SVM_Model.pkl')
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    del model
    from evaluation import evaluate
    return  evaluate(testtags, y_pred,y_score)
def randomforest(trainsdata,traintags,testsdata,testtags):
    from sklearn.ensemble import RandomForestClassifier
    # print ("RandomForestClassifier")
    rf0 = RandomForestClassifier(oob_score=True)
    # print ("Training")
    rf0.fit(trainsdata,traintags)
    y_pred = rf0.predict(testsdata)
    y_score = rf0.predict_proba(testsdata)

    return evaluate(testtags, y_pred,y_score)
def adaboost(trainsdata,traintags,testsdata,testtags):
    from sklearn.ensemble import AdaBoostClassifier
    # print("AdaBoostClassifier")
    clf = AdaBoostClassifier(n_estimators=100,learning_rate=0.1)
    # print("Training")
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
def lightgbm(traindata,traintags,testdata,testtags):

    clf = LGBMClassifier()
    # print("Training")
    clf.fit(traindata, traintags)
    train_label = clf.predict(traindata)
    y_pred = clf.predict(testdata)
    y_score = clf.predict_proba(testdata)

    return evaluate(testtags, y_pred,y_score)

from evaluation import evaluateLight
def catboost_cl(traindata,traintags,testdata,testtags):
    clf = CatBoostClassifier(logging_level="Silent")
    #print("Training")
    clf.fit(traindata,traintags)
    train_label = clf.predict(traindata)
    y_pred = clf.predict(testdata)
    y_score = clf.predict_proba(testdata)


    return evaluate(testtags, y_pred,y_score)
import os

def catboost_TrainStag(traindata,traintags,testdata,testtags,feature_name="no_name", CV=False, data_num="no_num",  split="no_split"):
    model_path = "Acr_Catboost_Model/Acr_Catboost_Model_"+feature_name

    #clf = CatBoostClassifier(num_boost_round=1000,logging_level="Silent")
    clf = CatBoostClassifier(num_boost_round=10,thread_count=4,logging_level="Silent")
    # if(CV==True):
    #     model_cv_path = "Acr_CV_Catboost_Model/" + "data_" + str(data_num) + "/split_" + str(split)
    #     model_cv_name = model_cv_path + "/Acr_CV_Catboost_Model_" + feature_name
    #     mkd(model_cv_path)
    #     if(os.path.exists(model_cv_name)):
    #         print(model_cv_name+" existed")
    #         clf.load_model(model_cv_name)
    #     else:
    #         print(model_cv_name+" has not existed")
    #         clf.fit(traindata,traintags)
    #         clf.save_model(model_cv_name)
    # else:
    #
    #     if(os.path.exists(model_path)):
    #         print("hello")
    #         #print(model_path+" existed")
    #         clf.load_model(model_path)
    #     else:
    #         #print(model_path+" has not existed")
    #         clf.fit(traindata,traintags)
    #         clf.save_model(model_path)
    clf.fit(traindata, traintags)
    train_label = clf.predict(traindata)
    y_pred = clf.predict(testdata)
    y_score = clf.predict_proba(testdata)
    return (train_label, y_pred, evaluate(testtags, y_pred, y_score))
def lightgbmTrainStag_old(traindata,traintags,testdata,testtags,params):
    model_path="light_model\\"

    train_data=lgb.Dataset(traindata,label=traintags,silent=True)
    # validation_data=lgb.Dataset(testdata,label=testtags)
    clf=lgb.train(params,train_data)
    # clf.save_model(model_path+"lightgbm_model"+method+".txt")

    train_label=clf.predict(traindata,predict_disable_shape_check=True,num_iteration=clf.best_iteration)
    y_pred=clf.predict(testdata,num_iteration=clf.best_iteration,predict_disable_shape_check=True)

    # print y_raw
#    y_score = y_pred
#    print y_score
    for i in range(len(train_label)):
        if train_label[i]>0.5:train_label[i]=1
        else:train_label[i]=0
    for i in range(len(y_pred)):
        if y_pred[i]>0.5:y_pred[i]=1
        else:y_pred[i]=0

    #return (train_label,y_pred, evaluateLight(testtags,y_pred,y_raw))
    return (train_label,y_pred)
import joblib
def lightgbmTrainStag(traindata,traintags,testdata,testtags,params=1,method_name="No_name"):

    import  pickle
    clf = LGBMClassifier()
    #print("Training")
    clf.fit(traindata,traintags)
    joblib.dump(clf, 'model/LG_Model'+method_name+'.pkl')
    train_label = clf.predict(traindata)
    y_pred = clf.predict(testdata)
    y_score = clf.predict_proba(testdata)
    print(method_name)
    print(traindata.shape)

#    pickle.dumps(clf,'model/LG_Model'+method_name+'.pickle')
    return (train_label, y_pred, evaluate(testtags, y_pred, y_score))
#filepath=['featurefiles/neg-test.a.3.1.1.fasta.csv','featurefiles/neg-train.a.3.1.1.fasta.csv','featurefiles/pos-test.a.3.1.1.fasta.csv','featurefiles/pos-train.a.3.1.1.fasta.csv',]
#methodset = ["svm","randomforest","gradientboosting","lightgbm","xgboost","decisiontree","bayes","adaboost"]
#filepath=['featurefiles/neg-test.a.3.1.1.fasta.csv','featurefiles/neg-train.a.3.1.1.fasta.csv','featurefiles/pos-test.a.3.1.1.fasta.csv','featurefiles/pos-train.a.3.1.1.fasta.csv',]
filepath=['featurefiles/neg-test.txt.csv','featurefiles/neg-train.txt.csv','featurefiles/pos-test.txt.csv','featurefiles/pos-train.txt.csv',]
