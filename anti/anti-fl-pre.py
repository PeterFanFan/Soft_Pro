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

    tmpdir  ="E:\\data\\2019-data\\features"

    suffix = "csv"
    postrain = glob.glob(tmpdir + '/bmark-positive80*')
    negtrain = glob.glob(tmpdir + '/bmark-negative80*')
    postest = glob.glob(tmpdir + '/Ind-positive*')
    negtest = glob.glob(tmpdir + '/Ind-negative*')


    filegroup = {}
    filegroup['postrain']= postrain
    filegroup['negtrain']= negtrain
    filegroup['postest']= postest
    filegroup['negtest']= negtest

    print("Loading Feature")
    filepath = "Result/Independent_Metric_First_Layer/anti-cancer-lightgbm-rank.csv"
#    filepath = "Result/Independent_Metric_First_Layer/anti-cancer-catboost-rank.csv"
    method = Catboost_Rank_file(filepath)

    Svm_Acc_List = []
    Svm_Auc_List = []

    for feature_num in range(1,20):
        file = "Aver_Result_Lightgbm_Anti_Cancer" + str(feature_num) + ".csv"
        print("Feature num: ", feature_num)
        # file = "Aver_Result_" + str(feature_num) + ".csv"
        #file = "Aver_Result_Rank_Catboost" + str(feature_num) + ".csv"
        filepath = "Result\\Independent_Metric_Second_Layer\\" + file
        if(os.path.exists(filepath)):
            result = pd.read_csv(filepath, index_col=0, header=0)
        else:
            method_list = method[0:feature_num]
            print(method_list)
            datadics = datadic_nomean(filegroup, method_list)
            data = trainmodel_GBM(datadics)
            #data = trainmodel_CatBoost(datadics)
            result = model_Predict(data)
            result.to_csv(filepath)
            # datadics = datadic(filegroup,method_list)
            # datadics_list = generate_data_group(datadics)

        print(result)
        svm_acc_tmp = result["Svm"]["acc"]
        svm_auc_tmp = result["Svm"]["auc"]
        Svm_Acc_List.append(svm_acc_tmp)
        Svm_Auc_List.append(svm_auc_tmp)
        if(feature_num>5):
            Svm_Temp_List = Svm_Acc_List[feature_num-6:feature_num]
            if(Svm_Temp_List.index(max(Svm_Temp_List))==0):
                num = feature_num-5
                print(num)
                print(method[0:num])
                print(max(Svm_Temp_List))
                print("Acc has declined 5 steps")
                break


if __name__ == '__main__':
    main()
