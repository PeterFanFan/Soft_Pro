# -*- coding: UTF-8 -*-
from bayesHyperTuning import opt
from testShuffle import split_data
import pandas as pd
import time
import os
from sklearn.metrics import accuracy_score
from skfeature.function.sparse_learning_based import ls_l21
from sklearn.svm import SVC
from skfeature.utility.sparse_learning import construct_label_matrix_pan, feature_ranking
import glob
from core_function import *
import pandas as pd
from testmetric import *
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
# from bayesHyperTuning import opt
from cls import randomforest, bayes, logisticregression, decisiontree, adaboost, knn, svm_opt, catboost_TrainStag,catboost_cl
from cls import lightgbm
from cls import lightgbmTrainStag
from cls import svmOpt
from cls import trainmodel_multipleTags
from SVM_BayesHyperTuning import opt
from sklearn.metrics import roc_auc_score
from testShuffle import Ranking_Acc
def main():


    train_tmpdir = "E:\\AcrData\\NAR_PaCRISPR_Datasets\\PaCRISPR_Training_Dataset\\new-method-feature"
    test_tmpdir = "E:\\AcrData\\NAR_PaCRISPR_Datasets\\PaCRISPR_Independent_Dataset\\new-method-feature"
    case_study_tmpdir = "E:\\AcrData\\NAR_PaCRISPR_Datasets\\PaCRISPR_Case_Study\\new-method-feature"

    postrain = glob.glob(train_tmpdir + '/PaCRISPR_Training_Positive_98*')
    negtrain = glob.glob(train_tmpdir + '/PaCRISPR_Training_Negative_902*')
    postest = glob.glob(test_tmpdir + '/PaCRISPR_Independent_Positive_26*')
    negtest = glob.glob(test_tmpdir + '/PaCRISPR_Independent_Negative_260*')


    filegroup = {}
    filegroup['postrain']= postrain
    filegroup['negtrain']= negtrain
    filegroup['postest']= postest
    filegroup['negtest']= negtest

    print("Loading Feature")
    #filepath = "Catboost_Rank/Catboost_Ave_Rank.csv"
    filepath = "Result/Independent_Metric_First_Layer/A_Ranking.csv"
    method = Catboost_Rank_file(filepath)

    Svm_Acc_List = []
    Svm_Auc_List = []
    for feature_num in range(3,4):
        print("Feature num: ", feature_num)
        # file = "Aver_Result_" + str(feature_num) + ".csv"
        file = "Aver_Result_Lightgbm_" + str(feature_num) + ".csv"
        #file = "Aver_Result_Rank_Catboost" + str(feature_num) + ".csv"
        filepath = "Result\\Independent_Metric_Second_Layer\\" + file
        # if(os.path.exists(filepath)):
        #     print ("aaa")
        #     #result = pd.read_csv(filepath, index_col=0, header=0)
        # else:
        method_list = list(method[0:feature_num])
        print(type(method_list))
        print(method_list)

        method_list.append("DictionaryLearning.csv")
        print(method_list)
        datadics = datadic(filegroup,method_list)
        datadics_list = generate_data_group(datadics)
        result = train_GBM_data_group(datadics_list, feature_num)
            # result = train_Catboost_data_group(datadics_list, feature_num)
        print(result)
        svm_acc_tmp = result["Svm"]["acc"]
        svm_auc_tmp = result["Svm"]["auc"]
        Svm_Acc_List.append(svm_acc_tmp)
        Svm_Auc_List.append(svm_auc_tmp)
        if(feature_num>5):
            Svm_Temp_List = Svm_Acc_List[feature_num-6:feature_num]
            # print Svm_Temp_List
            # print(Svm_Acc_List[0])
            # print(Svm_Acc_List[1])
            # print(Svm_Temp_List.index(max(Svm_Temp_List)))
            if(Svm_Temp_List.index(max(Svm_Temp_List))==0 and Svm_Temp_List[0]!=Svm_Temp_List[1]):
                num = feature_num-5
                print(num)
                print(method[0:num])
                print(max(Svm_Temp_List))
                print("Acc has declined 5 steps")
                break


if __name__ == '__main__':
    main()
