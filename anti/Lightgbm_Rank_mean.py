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
from cls import randomforest, bayes, logisticregression, decisiontree, adaboost, knn, svm_opt, catboost_TrainStag
from cls import lightgbm
from cls import lightgbmTrainStag
from cls import svmOpt
from cls import trainmodel_multipleTags
from SVM_BayesHyperTuning import opt
from sklearn.metrics import roc_auc_score
from testShuffle import Ranking_Acc
def main():

    # train_tmpdir = "E:\\AcrData\\NAR_PaCRISPR_Datasets\\PaCRISPR_Training_Dataset\\new-method-feature"
    # test_tmpdir = "E:\\AcrData\\NAR_PaCRISPR_Datasets\\PaCRISPR_Independent_Dataset\\new-method-feature"
    # case_study_tmpdir = "E:\\AcrData\\NAR_PaCRISPR_Datasets\\PaCRISPR_Case_Study\\new-method-feature"

    # postrain = glob.glob(train_tmpdir + '/PaCRISPR_Training_Positive_98*')
    # negtrain = glob.glob(train_tmpdir + '/PaCRISPR_Training_Negative_902*')
    # postest = glob.glob(test_tmpdir + '/PaCRISPR_Independent_Positive_26*')
    # negtest = glob.glob(test_tmpdir + '/PaCRISPR_Independent_Negative_260*')

    tmpdir  ="E:\\data\\2019-data\\features"

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
    method_new = generateNew_Method_Name(step=2, min_len=6)
    method_old = ["-DT.csv", "-PDT-Profile.csv", "-Top-n-gram.csv", "-PSSM-DT.csv", "-CC-PSSM.csv", "-AC-PSSM.csv",
              "ACC-PSSM.csv", "kmer", "feature-AC.csv", "ACC.csv", "feature-CC.csv", "DP.csv", "DR.csv",
              "PC-PseAAC.csv", "PC-PseAAC-General.csv", "PDT.csv", "SC-PseAAC.csv", "SC-PseAAC-General.csv"]
    method_list = method_old + method_new
    datadics = datadic(filegroup, method_list)
    datadics_list = generate_data_group(datadics)
    train_Lightgbm_data_group_Rank(datadics_list, data_name="Anti_Cancer_2019")

if __name__ == '__main__':
    main()
