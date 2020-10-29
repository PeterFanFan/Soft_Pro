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
from cls import  lightgbm, svm_opt
from cls import  svmOpt
from cls import  lightgbmTrainStag

def main():
    filegroup = {}
    train_tmpdir = "E:\\AcrData\\NAR_PaCRISPR_Datasets\\PaCRISPR_Training_Dataset\\new-method-feature"
    test_tmpdir = "E:\\AcrData\\NAR_PaCRISPR_Datasets\\PaCRISPR_Independent_Dataset\\new-method-feature"
    case_study_tmpdir = "E:\\AcrData\\NAR_PaCRISPR_Datasets\\PaCRISPR_Case_Study\\new-method-feature"

    postrain = glob.glob(train_tmpdir + '/PaCRISPR_Training_Positive_98*')
    negtrain = glob.glob(train_tmpdir + '/PaCRISPR_Training_Negative_902*')

    filegroup = {}
    filegroup['postrain'] = postrain
    filegroup['negtrain'] = negtrain

    from core_function import Catboost_Rank_file
    filepath = "Catboost_Rank/Catboost_Ave_Rank.csv"

    method = Catboost_Rank_file(filepath)
    filepath = "Result\\CV\\Second_layer"

    k = 5
    i = 10
    count_split = 1
    m = 10
    le = 196
    KF = KFold(n_splits=k, shuffle=True, random_state=i)
    for feature_num in range(5, 6):
        print("Feature_num: ", feature_num)
        Aver_Result = []
        for r_state in range(m):
            print(r_state)
            # print("Random_Sample R_state: ", r_state)

            cvdatadics = {}
            metricList = []
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


            datadics = datadic(filegroup, method[0:feature_num],r_state)

            count_split = 1
            filepath = "Result\\CV\\Second_layer"
            file = filepath + "\\Aver_CV_Result_Rank_Catboost_Random_State_"+ str(r_state)+"_Feature_Num"+ str(feature_num) + ".csv"
            if (os.path.exists(file)):
                outcome = pd.read_csv(file, index_col=0, header=0)
            else:
                for train_index, test_index in KF.split(range(0,le)):
                    # print("split:", count_split)
                    for methodname in datadics.keys():
                        data = datadics[methodname]
                        selected_features = data[0]
                        traintags = data[1]
                        X_train, X_test = np.array(selected_features)[train_index], np.array(selected_features)[test_index]
                        Y_train, Y_test = np.array(traintags)[train_index], np.array(traintags)[test_index]
                        newdata = [X_train,Y_train,X_test,Y_test]
                        cvdatadics[methodname] = newdata
                    from cls import svm_old_opt
                    from core_function import trainmodel_CatBoost
                    newfeature = trainmodel_GBM(cvdatadics)
                    #newfeature = trainmodel_CatBoost(cvdatadics, CV=True)
                    # model = svm_best_parameters_cross_validation(newfeature[0], newfeature[1])
                    # indemetric = svm_old_opt(newfeature[2], newfeature[3], model)
                    indemetric = svm_opt(newfeature[0],newfeature[1],newfeature[2], newfeature[3])
                    #indemetric = svmOpt(newfeature[0],newfeature[1],newfeature[2], newfeature[3])
                    random_inde =randomforest(newfeature[0],newfeature[1],newfeature[2], newfeature[3])
                    bayes_inde =bayes(newfeature[0],newfeature[1],newfeature[2], newfeature[3])
                    logisticregression_inde =logisticregression(newfeature[0],newfeature[1],newfeature[2], newfeature[3])
                    adaboost_inde =adaboost(newfeature[0],newfeature[1],newfeature[2], newfeature[3])
                    knn_inde = knn(newfeature[0],newfeature[1],newfeature[2], newfeature[3])
                    decisiontree_inde = decisiontree(newfeature[0],newfeature[1],newfeature[2], newfeature[3])
                    # print(indemetric)
                    split.append(indemetric)
                    split_random.append(random_inde)
                    split_bayes.append(bayes_inde)
                    split_lg.append(logisticregression_inde)
                    split_ada.append(adaboost_inde)
                    split_knn.append(knn_inde)
                    split_decision_tree.append(decisiontree_inde)
                    indemetric = pd.DataFrame(indemetric)
                    col = ['acc', 'auc', 'sen', 'spec', 'mcc']
                    indemetric = indemetric.loc[0,col]
                    # print(indemetric)
                    count_split = count_split+1
                meandic = cvmean(split)
                meandic_random = cvmean(split_random)
                meandic_bayes = cvmean(split_bayes)
                meandic_lg = cvmean(split_lg)
                meandic_ada = cvmean(split_ada)
                meandic_knn = cvmean(split_knn)
                meandic_decision_tree = cvmean(split_decision_tree)
                metric = pd.DataFrame(meandic)
                meandic_random = pd.DataFrame(meandic_random)
                meandic_bayes = pd.DataFrame(meandic_bayes)
                meandic_lg = pd.DataFrame(meandic_lg)
                meandic_ada = pd.DataFrame(meandic_ada)
                meandic_knn = pd.DataFrame(meandic_knn)
                meandic_decision_tree = pd.DataFrame(meandic_decision_tree)
             #col = ['acc','auc','sen','spec','mcc','f1_score']
                col = ['acc', 'auc','f1_score', 'sen', 'spec', 'mcc']
                row = metric.loc[0, col]
                row_random = meandic_random.loc[0, col]
                row_bayes = meandic_bayes.loc[0, col]
                row_lg = meandic_lg .loc[0, col]
                row_ada = meandic_ada.loc[0, col]
                row_knn = meandic_knn.loc[0, col]
                row_decision_tree = meandic_decision_tree.loc[0, col]

                row.name = "Svm"
                row_random.name = "RF"
                row_bayes.name = "Bayes"
                row_lg.name = "LG"
                row_ada.name = "Adaboost"
                row_knn.name = "knn"
                row_decision_tree.name = "DT"
                o_list = [row, row_random, row_bayes, row_lg, row_ada, row_knn, row_decision_tree]
                outcome = pd.concat(o_list,axis=1)
                outcome.to_csv(file)
            Aver_Result.append(outcome)
            #print(outcome)
        Ave_R = get_Av_Result(Aver_Result, m)
        Ave_R.to_csv(filepath+"/Aver_CV_Result_Feature_Num_"+str(feature_num)+".csv")
def get_Av_Result(Aver_Result, m):
    print("Aver_Result")
    Aver_ = Aver_Result[0]
    #print(Aver_Result)
    for i in range(1, m):
        Aver_ = Aver_ + Aver_Result[i]
    Aver_ = Aver_ / m
    print(Aver_)
    return Aver_


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
def dataprocessingCV(filepath, methodname, k=0):
    # print ("Loading feature files")
    from  core_function import generateNew_Method_Name
    if methodname in generateNew_Method_Name():

        dataset2 = pd.read_csv(filepath[0], header=0, index_col=0, low_memory=False)
        dataset4 = pd.read_csv(filepath[1], header=0, index_col=0, low_memory=False)
    else:

    #print (filepath)
        dataset2 = pd.read_csv(filepath[0], header=None, low_memory=False)
        dataset4 = pd.read_csv(filepath[1], header=None, low_memory=False)
    # dataset1 = pd.read_csv('neg-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset2 = pd.read_csv('neg-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset3 = pd.read_csv('pos-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset4 = pd.read_csv('pos-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset1=pd.DataFrame(dataset1,dtype=np.float)
    # print ("Feature processing")
    # dataset1 = dataset1.convert_objects(convert_numeric=True)
    dataset2 = dataset2.convert_objects(convert_numeric=True)
    # dataset3 = dataset3.convert_objects(convert_numeric=True)
    dataset4 = dataset4.convert_objects(convert_numeric=True)
    # dataset1.dropna(inplace = True)
    dataset2.dropna(inplace=True)
    # dataset3.dropna(inplace = True)
    dataset4.dropna(inplace=True)
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.ensemble import EasyEnsemble
    # 建立模型
    under_model = RandomUnderSampler(random_state=k)
    traindata = pd.concat([dataset2, dataset4], axis=0)
    negtraintags = [0] * dataset2.shape[0]
    postraintags = [1] * dataset4.shape[0]
    traintags = negtraintags + postraintags
    traindata, traintags = under_model.fit_sample(traindata, traintags)

    # print(len(traintags))
    data = [traindata, traintags]
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
def datadic(filegroup, method, CV_rs):
    #method = ["-PSSM-RT.csv","-PSSM-DT.csv","CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","fastafeature-AC.csv","ACC.csv","fastafeature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    #method = ["-PSSM-DT.csv"]
    # method = ["-DT.csv","-PDT-Profile.csv","-Top-n-gram.csv","-PSSM-RT.csv","-PSSM-DT.csv","-CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    # #method = ["-DT.csv","-PDT-Profile.csv","-Top-n-gram.csv","-CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    #
    # method = ["-DT.csv","-PDT-Profile.csv","kmer"]
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
        # print (filepath)
        file_method[methodname] = dataprocessingCV(filepath, i,k=CV_rs)
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