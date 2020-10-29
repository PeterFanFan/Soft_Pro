#-*- coding: UTF-8 -*-
import warnings
from sklearn.metrics import auc
from scipy import interp
# from cls import dataprocessing
from cls import xgboost
from cls import lightgbm
import xgboost as xgbs
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

# import warnings
# warnings.filterwarnings("ignore")
def cvmean(dics):

    meandic = {}
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for dic in dics:
        for i in dic:
            if meandic:
                if (i is 'rocs')==False:
                    meandic[i] = dic[i]+meandic[i]
                # if (isinstance(dic[i],list)) == False:
                #     meandic[i] = dic[i]+meandic[i]

                else:
                    #print dic[i]
                    # for m in range(0,len(dic[i])):
                    #     temp = dic[i][m]+meandic[i][m]
                    fpr = dic[i][0]
                    tpr = dic[i][1]
                    tprs.append(interp(mean_fpr, fpr, tpr))


            else:
                meandic = dic
                fpr = dic['rocs'][0]
                tpr = dic['rocs'][1]
                tprs.append(interp(mean_fpr, fpr, tpr))
                break

    for i in meandic:
        if (isinstance(meandic[i], list)) == False:
            meandic[i] = meandic[i]/len(dics)
            # print(i,meandic[i])
        else:
            pass
            # for m in range(0,len(meandic)):
            #     meadic[i][m] = np.mean(meandic[i][m])
            # a = meandic[i]
            # meanlis = [a[j]/k for j in range(len(a))]
            # meandic[i] = meanlis
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # meandic['auc'] = mean_auc
    meandic['rocs'][0]= mean_fpr
    meandic['rocs'][1]= mean_tpr

    return meandic
def cvtest(traindata, traintags,params):
    filepath=['featurefiles/neg-test.txt.csv','featurefiles/neg-train.txt.csv','featurefiles/pos-test.txt.csv','featurefiles/pos-train.txt.csv',]
    k = 5   #k折
    KF = KFold(n_splits = k,shuffle=True,random_state=10)
    #data为数据集,利用KF.split划分训练集和测试集
    # traindata, traintags, testdata, testtags = dataprocessing(filepath)
    count = 1
    data = {}
    meandic = {}
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for train_index,test_index in KF.split(traindata):
        #建立模型，并对训练集进行测试，求出预测得分
        #划分训练集和测试集
        #print test_index
        X_train,X_test = np.array(traindata)[train_index],np.array(traindata)[test_index]
        Y_train,Y_test = np.array(traintags)[train_index],np.array(traintags)[test_index]
        #建立模型(模型已经定义)
        # model = build_model()
        # #编译模型
        # model.compile(optimizer = 'sgd',loss = 'categorical_crossentropy',metrics = ['acc'])
        # #训练模型
        # model.fit(X_train,Y_train,batch_size = 2,validation_data = (X_test,Y_test),epochs = 150)
        #利用model.predict获取测试集的预测值
        dic = xgboost(X_train,Y_train,X_test,Y_test,params)
        for i in dic:
            if meandic:
                if (i is 'rocs')==False:
                    meandic[i] = dic[i]+meandic[i]
                # if (isinstance(dic[i],list)) == False:
                #     meandic[i] = dic[i]+meandic[i]

                else:
                    #print dic[i]
                    # for m in range(0,len(dic[i])):
                    #     temp = dic[i][m]+meandic[i][m]
                    fpr = dic[i][0]
                    tpr = dic[i][1]
                    tprs.append(interp(mean_fpr, fpr, tpr))


            else:
                meandic = dic
                fpr = dic['rocs'][0]
                tpr = dic['rocs'][1]
                tprs.append(interp(mean_fpr, fpr, tpr))
                break
        #data['xg'+str(count)] = dic
        count = count+1
        #计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
        # fpr,tpr,thresholds=roc_curve(Y_test[:,1],y_pred[:,1])
        # #interp:插值 把结果添加到tprs列表中
        # tprs.append(interp(mean_fpr,fpr,tpr))
        # tprs[-1][0]=0.0
        # #计算auc
        # roc_auc=auc(fpr,tpr)
        # aucs.append(roc_auc)
        # #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
        # plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i,roc_auc))
        # i +=1

    for i in meandic:
        if (isinstance(meandic[i], list)) == False:
            meandic[i] = float(meandic[i])/k
            # print(i,meandic[i])
        else:
            pass
            # for m in range(0,len(meandic)):
            #     meadic[i][m] = np.mean(meandic[i][m])
            # a = meandic[i]
            # meanlis = [a[j]/k for j in range(len(a))]
            # meandic[i] = meanlis
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    meandic['auc'] = mean_auc
    meandic['rocs'][0]= mean_fpr
    meandic['rocs'][1]= mean_tpr

    # data['xgmean'] = meandic
    #
    # rocauc(data,'Cross Validation')
    return meandic
def indetest(traindata, traintags, testsdata, testtags,params):
        # selected_features = traindata[:, idx[0:num_fea]]
        # selected_featurestest = testsdata[:, idx[0:num_fea]]
        # alg.fit(selected_features,traintags, eval_metric='auc')
        # Predict training set:
        # dtrain_predictions = alg.predict(dtrain)
        # dtrain_predprob = alg.predict_proba(dtrain)[:, 1]

        # dtest_predictions = alg.predict(selected_featurestest)
        # dtest_predprob = alg.predict_proba(selected_featurestest)[:, 1]
        meandic = lightgbm(traindata,traintags, testsdata, testtags, params)
        print('acc:', meandic['acc'], "auc", meandic['auc'], "mcc", meandic['mcc'], "sen", meandic['sen'], "spec",
              meandic['spec'])
        return meandic
        # print "\nModel Report"
        # print "Train Accuracy : %.4g" % metrics. accuracy_score(dtraintags, dtrain_predictions)
        # print "Train AUC Score (Train )(use probalility): %f" % metrics.roc_auc_score(dtraintags, dtrain_predprob)
        # print "Train AUC Score (Train)(use tags): %f" % metrics.roc_auc_score(dtraintags, dtrain_predictions)

        # print "Test Accuracy : %.4g" % metrics.accuracy_score(testtags, dtest_predictions)
        # print "Test AUC Score (Test)(use probalility): %f" % metrics.roc_auc_score(testtags, dtest_predprob)

        # print "Test AUC Score (Test)(use tags): %f" % metrics.roc_auc_score(testtags, dtest_predictions)
def paraoptimize(selected_features,traintags):
    KF = KFold(n_splits=10, shuffle=True, random_state=10)
        # number of selected features
    correct = 0
    # selected_features = traindata[:, idx[0:num_fea]]
    # selected_featurestest = testsdata[:, idx[0:num_fea]]
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'gamma': 0,
              'random_state': 50,
              'learning rate': 0.1,
              'max_depth': 3,
              'lambda': 10,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'min_child_weight': 1,
              'eta': 0.025,
              'seed': 27,
              'nthread': -1,
              'scale_pos_weight': 1,
              'silent': 0}

    xgb = XGBClassifier(
        booster='gbtree',
     learning_rate =0.1,
     n_estimators=100,
     max_depth=3,
     lamba = 10,
     # reg_alpha = 1e-5,
     min_child_weight=1,
     gamma=0,
     # eval_metric='rmse',

    random_state=50,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=-1,
     eta=0.025,
     scale_pos_weight=1,
     seed=27,
     silent=1)
    # xgb.se
    # xgb.set_params(params)


    # def (alg):
    #     alg.fit(selected_features,traintags, eval_metric='auc')
    #     # Predict training set:
    #     # dtrain_predictions = alg.predict(dtrain)
    #     # dtrain_predprob = alg.predict_proba(dtrain)[:, 1]
    #
    #     dtest_predictions = alg.predict(selected_featurestest)
    #     dtest_predprob = alg.predict_proba(selected_featurestest)[:, 1]
    #
    #     print "\nModel Report"
    #     # print "Train Accuracy : %.4g" % metrics. accuracy_score(dtraintags, dtrain_predictions)
    #     # print "Train AUC Score (Train )(use probalility): %f" % metrics.roc_auc_score(dtraintags, dtrain_predprob)
    #     # print "Train AUC Score (Train)(use tags): %f" % metrics.roc_auc_score(dtraintags, dtrain_predictions)
    #
    #     print "Test Accuracy : %.4g" % metrics.accuracy_score(testtags, dtest_predictions)
    #     print "Test AUC Score (Test)(use probalility): %f" % metrics.roc_auc_score(testtags, dtest_predprob)
    #     print "Test AUC Score (Test)(use tags): %f" % metrics.roc_auc_score(testtags, dtest_predictions)

    def modelfit(alg, dtrain,  dtraintags,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):


        if useTrainCV:

            xgb_param = alg.get_xgb_params()

            xgtrain = xgbs.DMatrix(dtrain, label=dtraintags)
            cvresult = xgbs.cv(xgb_param, xgtrain, num_boost_round=5000, nfold=cv_folds,
                              metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=False,seed = 10)
            alg.set_params(n_estimators=cvresult.shape[0])
        cur = cvresult.shape[0]
        res = cvresult.shape[1]
        #print(res)
        #print(cvresult)
        #print(type(cvresult))
        # Fit the algorithm on the data



        return cur# 以交叉验证作为评价方法，寻找出最优的迭代次数
    rounds = modelfit(xgb, selected_features,traintags)

    param_test1 = {
     'max_depth':range(1,10,1),
     'min_child_weight':range(1,6,1)
    }
    gsearch1 = GridSearchCV(estimator = xgb.set_params(n_estimators=rounds),
    param_grid = param_test1,scoring='accuracy',n_jobs=-1,iid=True, cv=KF)
    gsearch1.fit(selected_features,traintags)
    #print(gsearch1.cv_results_ )
    #print(gsearch1.best_params_)
    #print(gsearch1.best_score_)
    bestpara = gsearch1.best_params_
    min_c = xgb.get_params()['min_child_weight']
    max_dep = xgb.get_params()['max_depth']
    lis = [max_dep,min_c]
    bestvalue = bestpara.values
    if(cmp(bestvalue,lis)==0):
        pass
    else:
        xgb.set_params(max_depth=bestpara['max_depth'],min_child_weight=bestpara['min_child_weight'])
        
        rounds = modelfit(xgb, selected_features, traintags)


    # param_test2 = {
    #  'max_depth':[2,3,4],
    #  'min_child_weight':[1,2]
    # }
    # gsearch2 = GridSearchCV(estimator = XGBClassifier(         learning_rate =0.1, n_estimators=110, max_depth=5,
    # min_child_weight=1, gamma=0, subsample=0.8,             colsample_bytree=0.8,
    #  objective= 'binary:logistic', nthread=-1,     scale_pos_weight=1, seed=27),
    #  param_grid = param_test2,     scoring='accuracy',n_jobs=4,iid=True, cv=5)
    # gsearch2.fit(selected_features,traintags)
    # print(gsearch2.cv_results_ )
    # print(gsearch2.best_params_)
    # print(gsearch2.best_score_)
    # 参数已改变，重新选择最佳迭代次序
    # xgb2 = XGBClassifier(
    #  learning_rate =0.1,
    #  n_estimators=1000,
    #  max_depth=3,
    #  min_child_weight=1,
    #  gamma=0,
    #  subsample=0.8,
    #  colsample_bytree=0.8,
    #  objective= 'binary:logistic',
    #  nthread=-1,
    #  scale_pos_weight=1,
    #  seed=27)r
    # modelfit(xgb2, selected_features,traintags,selected_featurestest,testtags)
    param_test3 = {
     'gamma':[i/10.0 for i in range(0,5)]
    }
    gsearch3 = GridSearchCV(estimator = xgb.set_params(n_estimators=rounds),
    param_grid = param_test3, scoring='accuracy',n_jobs=-1,iid=True, cv=KF)
    gsearch3.fit(selected_features,traintags)
    #print(gsearch3.cv_results_ )
    #print(gsearch3.best_params_)
    #print(gsearch3.best_score_)
    bestpara = gsearch3.best_params_
    gam = xgb.get_params()['gamma']
    lis = [gam]
    bestvalue = bestpara.values
    if (cmp(bestvalue, lis) == 0):
        pass
    else:
        xgb.set_params(gamma=bestpara['gamma'])
        (xgb)
        rounds = modelfit(xgb, selected_features, traintags)
    param_test4 = {
     'subsample':[i/10.0 for i in range(6,10)],
     'colsample_bytree':[i/10.0 for i in range(6,10)]
    }

    gsearch4 = GridSearchCV(estimator = xgb.set_params(n_estimators=rounds),
    param_grid = param_test4, scoring='accuracy',n_jobs=-1,iid=True, cv=KF)

    gsearch4.fit(selected_features,traintags)
    #print(gsearch4.cv_results_ )
    #print(gsearch4.best_params_)
    #print(gsearch4.best_score_)
    bestpara = gsearch4.best_params_
    col = xgb.get_params()['colsample_bytree']
    sub = xgb.get_params()['subsample']
    lis = [sub,col]
    bestvalue = bestpara.values
    if (cmp(bestvalue, lis) == 0):
        pass
    else:

        xgb.set_params(colsample_bytree=bestpara['colsample_bytree'],subsample=bestpara['subsample'])
        rounds = modelfit(xgb, selected_features, traintags)

    # param_test5 = {
    #  'subsample':[i/100.0 for i in range(75,90,5)],
    #  'colsample_bytree':[i/100.0 for i in range(75,90,5)]
    # }
    # gsearch5 = GridSearchCV(estimator = XGBClassifier(         learning_rate =0.1, n_estimators=168, max_depth=3,
    # min_child_weight=1, gamma=0, subsample=0.8,             colsample_bytree=0.8,
    #  objective= 'binary:logistic', nthread=-1,     scale_pos_weight=1, seed=27),
    #  param_grid = param_test5,     scoring='accuracy',n_jobs=4,iid=True, cv=5)
    # gsearch5.fit(selected_features,traintags)
    # print(gsearch5.cv_results_ )
    # print(gsearch5.best_params_)
    # print(gsearch5.best_score_)

    param_test6 = {
     'reg_alpha':[1e-5, 1e-2, 0.1, 1,100,1e-6]
    }
    gsearch6 = GridSearchCV(estimator = xgb.set_params(n_estimators=rounds),
    param_grid = param_test6, scoring='accuracy',n_jobs=-1,iid=True, cv=KF)
    gsearch6.fit(selected_features,traintags)
    #print(gsearch6.cv_results_ )
    # print(gsearch6.best_params_)
    # print(gsearch6.best_score_)
    bestpara = gsearch6.best_params_
    reg = xgb.get_params()['reg_alpha']
    lis = [reg]
    bestvalue = bestpara.values
    if(cmp(bestvalue,lis)==0):
        pass
    else:
        xgb.set_params(reg_alpha=bestpara['reg_alpha'])

        #rounds = modelfit(xgb, selected_features, traintags, selected_featurestest, testtags)
    # param_test7 = {
    #  'reg_alpha':[0.00001, 0.00002, 0.00003, 0.000009, 0.000008]
    # }
    # gsearch7= GridSearchCV(estimator = XGBClassifier(         learning_rate =0.1, n_estimators=168, max_depth=3,
    # min_child_weight=1, gamma=0, subsample=0.8,             colsample_bytree=0.8,
    #  objective= 'binary:logistic', nthread=-1,     scale_pos_weight=1, seed=27),
    #  param_grid = param_test7,     scoring='accuracy',n_jobs=4,iid=True, cv=5)
    # gsearch7.fit(selected_features,traintags)
    # print(gsearch7.cv_results_ )
    # print(gsearch7.best_params_)
    # print(gsearch7.best_score_)
    # file = r'parameter.txt'
    # selectmethod = 'ls'
    # selectnumber = 'featurenum '+i
    # para         = str(xgb.get_params())
    # with open(file, 'a+') as f:
    #     f.write('/n' + selectmethod+selectnumber+para)
    params = xgb.get_params()
    return params
    #
    #
    # (xgb)
def cvt(selected_features,traintags,params):
    split = []
    k =5   #k折
    KF = KFold(n_splits = k,shuffle=True,random_state=10)
    for train_index, test_index in KF.split(selected_features):
        X_train, X_test = np.array(selected_features)[train_index], np.array(selected_features)[test_index]
        Y_train, Y_test = np.array(traintags)[train_index], np.array(traintags)[test_index]
        # params = paraoptimize(X_train,Y_train)
        it = indetest(X_train, Y_train, X_test, Y_test, params)
        split.append(it)

    meandic = cvmean(split)
    col = ['acc','auc','sen','spec','mcc','f1_score']
    # print meandic.loc[:,col]
    return meandic
    # print('acc:', meandic['acc'], "auc", meandic['auc'], "mcc", meandic['mcc'], "sen", meandic['sen'], "spec",
    #       meandic['spec'])

