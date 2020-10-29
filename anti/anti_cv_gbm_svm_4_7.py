#-*- coding: UTF-8 -*-
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import svm
import os
from testmetric import *
from bayesHyperTuning import opt
import glob
from imblearn.over_sampling import SMOTE

import warnings
from cls import randomforest,bayes,logisticregression,decisiontree,adaboost,knn
from cls import  lightgbm
from cls import  svmOpt
from cls import  lightgbmTrainStag

def main():
    warnings.filterwarnings("ignore")
    print("hello world")
    tmpdir = "F:\\Programs\\journals_papers\\Anticancer Peptides\\anti-2020-3-27\\data\\AntiCAP_138pos_206neg_2014.321\\method-feature"
    tmpdir = "E:data\\AntiCAP_138pos_206neg_2014.321\\method-feature"

    #tmpdir = "E:\\data\ACP_2018_CV\\method-feature"
#    tmpdir = "F:\\Programs\\journals_papers\\Anticancer Peptides\\anti-2020-3-27\\data\\ACP_2018_CV\\method-feature"




    suffix  = "csv"
    postrain = glob.glob(tmpdir + '/*138*')
    negtrain = glob.glob(tmpdir + '/*206*')

    filegroup = {}
    train_tmpdir = "E:\\AcrData\\NAR_PaCRISPR_Datasets\\PaCRISPR_Training_Dataset\\new-method-feature"
    test_tmpdir = "E:\\AcrData\\NAR_PaCRISPR_Datasets\\PaCRISPR_Independent_Dataset\\new-method-feature"
    case_study_tmpdir = "E:\\AcrData\\NAR_PaCRISPR_Datasets\\PaCRISPR_Case_Study\\new-method-feature"

    postrain = glob.glob(train_tmpdir + '/PaCRISPR_Training_Positive_98*')
    negtrain = glob.glob(train_tmpdir + '/PaCRISPR_Training_Negative_902*')
    postest = glob.glob(test_tmpdir + '/PaCRISPR_Independent_Positive_26*')
    negtest = glob.glob(test_tmpdir + '/PaCRISPR_Independent_Negative_260*')
    filegroup = {}
    filegroup['postrain'] = postrain
    filegroup['negtrain'] = negtrain
    filegroup['postrain']= postrain
    filegroup['negtrain']= negtrain

    datadics = datadic(filegroup)
    le = len(datadics['kmer'][1])
    print(le)

    cvdatadics = {}
    ##138-206

    metricList=[]
    randomList = []
    bayesList = []
    lgList = []
    adaList = []
    knnList = []
    decisionList = []

    split = []
    split_random = []
    split_bayes = []
    split_lg = []
    split_ada = []
    split_knn = []
    split_decision_tree = []
    k=5
    m=31
    i = 10
    KF = KFold(n_splits=k, shuffle=True, random_state=i)
    for train_index, test_index in KF.split(range(0,le)):
        for methodname in datadics.keys():
            data = datadics[methodname]
            selected_features = data[0]
            traintags = data[1]
            X_train, X_test = np.array(selected_features)[train_index], np.array(selected_features)[test_index]
            Y_train, Y_test = np.array(traintags)[train_index], np.array(traintags)[test_index]
            newdata = [X_train,Y_train,X_test,Y_test]
            cvdatadics[methodname] = newdata
##svm opt
#        newfeature = trainmodel(cvdatadics)
##lightGBM opt
        newfeature = trainmodel_GBM(cvdatadics)

        model = svm_best_parameters_cross_validation(newfeature[0], newfeature[1])
        indemetric = svmOpt(newfeature[2], newfeature[3], model)
        random_inde =randomforest(newfeature[0],newfeature[1],newfeature[2], newfeature[3])
        bayes_inde =bayes(newfeature[0],newfeature[1],newfeature[2], newfeature[3])
        logisticregression_inde =logisticregression(newfeature[0],newfeature[1],newfeature[2], newfeature[3])
        adaboost_inde =adaboost(newfeature[0],newfeature[1],newfeature[2], newfeature[3])
        knn_inde = knn(newfeature[0],newfeature[1],newfeature[2], newfeature[3])
        decisiontree_inde = decisiontree(newfeature[0],newfeature[1],newfeature[2], newfeature[3])

        split.append(indemetric)
        split_random.append(random_inde)
        split_bayes.append(bayes_inde)
        split_lg.append(logisticregression_inde)
        split_ada.append(adaboost_inde)
        split_knn.append(knn_inde)
        split_decision_tree.append(decisiontree_inde)

    meandic = cvmean(split)
    meandic_random = cvmean(split_random)
    meandic_bayes = cvmean(split_bayes)
    meandic_lg = cvmean(split_lg)
    meandic_ada = cvmean(split_ada)
    meandic_knn = cvmean(split_knn)
    meandic_decision_tree = cvmean(split_decision_tree)


    print meandic['acc']
    metric = pd.DataFrame(meandic)
    meandic_random = pd.DataFrame(meandic_random)
    meandic_bayes = pd.DataFrame(meandic_bayes)
    meandic_lg = pd.DataFrame(meandic_lg)
    meandic_ada = pd.DataFrame(meandic_ada)
    meandic_knn = pd.DataFrame(meandic_knn)
    meandic_decision_tree = pd.DataFrame(meandic_decision_tree)
     #col = ['acc','auc','sen','spec','mcc','f1_score']
    col = ['acc','sen','spec','mcc']

    row = metric.loc[0, col]
    row_random = meandic_random.loc[0, col]
    row_bayes = meandic_bayes.loc[0, col]
    row_lg = meandic_lg .loc[0, col]
    row_ada = meandic_ada.loc[0, col]
    row_knn = meandic_knn.loc[0, col]
    row_decision_tree = meandic_decision_tree.loc[0, col]
    print(row)
    print(row_random)
    print(row_bayes)
    print(row_lg)
    print(row_ada)
    print(row_knn)
    print(row_decision_tree)
    #
    # row.name = i
    # row =row.to_frame()
    # # print row
    # metricList.append(row)
    # randomList.append(row_random)
    # bayesList.append(row_bayes)
    # lgList.append(row_lg)
    # adaList.append(row_ada)
    # knnList.append(row_knn)
    # decisionList.append(row_decision_tree)

    # res = pd.concat(metricList, axis=1).T
    # res_random = pd.concat(randomList, axis=1).T
    # res_bayes = pd.concat(bayesList, axis=1).T
    # res_lg = pd.concat(lgList, axis=1).T
    # res_ada = pd.concat(adaList, axis=1).T
    # res_knn = pd.concat(knnList, axis=1).T
    # res_decision_tree = pd.concat(decisionList, axis=1).T
    #
    # #res = res.T
    # import time
    # import matplotlib.pyplot as plt
    # nowtime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # file = "cv_test_"+nowtime+".csv"
    # filepath="Crossvalidation_metric\\"+file
    # res.to_csv(filepath)
    # res.plot(kind='box',title="Different partition of data",style='--')
    # #plt.xlabel("random_state")
    # plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    # plt.rcParams['figure.dpi'] = 300  # 分
    # plt.savefig("picture\\Box_4_15.png")
    # # plt.show()
    #
    # meanMetric_svm = res.mean(axis=0)
    # meanMetric_random = res_random.mean(axis=0)
    # meanMetric_bayes = res_bayes.mean(axis=0)
    # meanMetric_lg = res_lg.mean(axis=0)
    # meanMetric_ada = res_ada.mean(axis=0)
    # meanMetric_knn = res_knn.mean(axis=0)
    # meanMetric_decision_tree = res_decision_tree.mean(axis=0)
    #
    # meanMetric_svm.name="Svm"
    # meanMetric_random.name="Random_forest"
    # meanMetric_bayes.name="Bayes"
    # meanMetric_lg.name="LG"
    # meanMetric_ada.name="Adaboost"
    # meanMetric_knn.name="Knn"
    # meanMetric_decision_tree.name="Decision_tree"
    #
    # outList = [meanMetric_svm,meanMetric_random,meanMetric_bayes,meanMetric_lg,meanMetric_ada,meanMetric_decision_tree]
    # outCome = pd.concat(outList,axis=1)
    # filename = "Cv_result_compare.csv"
    # filepath = "Crossvalidation_metric\\"+filename
    # outCome.to_csv(filepath)
    # #meanMetric = meanMetric.to_frame()
    # print outCome
   # print metric.loc[0,col]

def svm_best_parameters_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(probability=False)
    #param_grid = {'kernel':('linear', 'rbf'),'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    param_grid = {'C': [1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000,10000,100000], 'gamma': [1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000,10000]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = -1, verbose=1,cv=10)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    print(best_parameters)
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model
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

    traindata = pd.concat([dataset2, dataset4],axis=0)
    smo = SMOTE(random_state=42)
    negtraintags = [0]*dataset2.shape[0]
    postraintags= [1]*dataset4.shape[0]
    traintags = negtraintags+postraintags
    traindata,traintags = smo.fit_sample(traindata, traintags)
    data = [traindata,traintags]
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

    traindata = pd.concat([dataset2, dataset4],axis=0)
    testdata = pd.concat([dataset1, dataset3])

    smo = SMOTE(random_state=42)

    negtraintags = [0]*dataset2.shape[0]
    postraintags= [1]*dataset4.shape[0]
    traintags = negtraintags+postraintags
    # testdata = pd.concat([dataset1, dataset3])
    negtesttags = [0]*dataset1.shape[0]
    postesttags= [1]*dataset3.shape[0]
    testtags = negtesttags+postesttags
    traindata,traintags = smo.fit_sample(traindata, traintags)
    data = [traindata,traintags,testdata,testtags]
    return data
#得到路径下所有数据编号
def matchfiles(tmpdir,suffix):  # 读取文件路径
    ###windows os
    # f = glob.glob(tmpdir + '\\*.' + suffix)
    ###linux os
    fi = []
    filenames = []
    #f = glob.glob(tmpdir + suffix)
    f = glob.glob(tmpdir + '/*.' + suffix)
    return f

def mkd(x):
    if(os.path.exists(x)):
        pass
    else:
        os.makedirs(x)

#合并编号对应阳阴训练集或测试集并保存
def datadic(filegroup):
    #method = ["-PSSM-RT.csv","-PSSM-DT.csv","CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","fastafeature-AC.csv","ACC.csv","fastafeature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    #method = ["-PSSM-DT.csv"]
    method = ["-DT.csv","-PDT-Profile.csv","-Top-n-gram.csv","-PSSM-RT.csv","-PSSM-DT.csv","-CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    #method = ["-DT.csv","-PDT-Profile.csv","-Top-n-gram.csv","-CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    method = ["-DT.csv","-PDT-Profile.csv"]
    #method = ["-PSSM-RT.csv","-PSSM-DT.csv","CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","fastafeature-AC.csv","ACC.csv","fastafeature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    #method = ["kmer","fastafeature-AC.csv"]

    postrain = filegroup["postrain"]
    negtrain = filegroup["negtrain"]
    file_method = {}
    filepath = []
    for methodname in method:
        for i in postrain:
            if methodname in i:
                postrain_method = i
                break
        #匹配出methodname对应的文件
        for j in negtrain:
            if methodname in j:
                negtrain_method = j
                break
        #匹配出methodname对应的文件

    # dataset1 = pd.read_csv('neg-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset2 = pd.read_csv('neg-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset3 = pd.read_csv('pos-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset4 = pd.read_csv('pos-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
        filepath = [negtrain_method,postrain_method]
        print (filepath)
        file_method[methodname] = dataprocessingCV(filepath)
        filepath = []
    return file_method

#训练11个模型
	# return model

#训练11个模型
def trainmodel_GBM(datadic):
    train_feature = {}
    test_feature = {}
    metrics = {}
    index = []
    metrics['Mcc'] = []
    metrics['Acc'] = []
    metrics['Sen'] = []
    metrics['Sp'] = []
    metrics['Auc'] = []
    for i in datadic:
        data = datadic[i]
        index.append(i)
        #        model = svm.SVC(probability=False)
        #        print("Svm training")
        #        print i
        #        print data[0].shape
        #        print data[2].shape
        #        model.fit(data[0], data[1])

        #        model=svm_best_parameters_cross_validation(data[0], data[1])
        cls = LGBMClassifier()
        params = cls.get_params()
        #        params = opt(data[0],data[1])

        #        indemetric = lightgbm(data[0],data[1],data[2],data[3],params)
        #        y_pred_train= model.predict(data[0])
        #        y_pred_test = model.predict(data[2])

        (y_pred_train, y_pred_test, metric) = lightgbmTrainStag(data[0], data[1], data[2], data[3], params)
        metrics['Mcc'].append(metric['mcc'])
        metrics['Acc'].append(metric['acc'])
        metrics['Sen'].append(metric['sen'])

        metrics['Sp'].append(metric['spec'])
        metrics['Auc'].append(metric['auc'])
        train_feature[i] = y_pred_train
        test_feature[i] = y_pred_test
        # data = [traindata,traintags,testdata,testtags]
    # import time
    outcome = pd.DataFrame(metrics, index=index)
    # nowtime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # file = "independent_test_" + nowtime + ".csv"
    # filepath = "Independent_metric\\" + file
    # outcome.to_csv(filepath)
    train_feature_vector = pd.DataFrame(train_feature)
    test_feature_vector = pd.DataFrame(test_feature)
    data[0] = train_feature_vector.values
    data[2] = test_feature_vector.values
    #    data[0] = train_feature_vector
    #    data[2] = test_feature_vector
    return data
	# return model


if __name__ == '__main__':
	main()