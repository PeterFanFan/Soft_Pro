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
import numpy as np
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
    filegroup['postrain'] = postrain
    filegroup['negtrain'] = negtrain
    filegroup['postest'] = postest
    filegroup['negtest'] = negtest

    print("Loading Feature")

    method = Ranking_Acc()
    lst1 = generateNew_Method_Name()
    new_method_list = []
    for method_name in method:
        if method_name in lst1:
            new_method_list.append(method_name)
            if "forward" in method_name:
                method_name_split = method_name[11:]
                lst2 = generateNew_Method_Name(encoding_types=[method_name_split,])
                lst1 =del_lst2_from_lst1(lst1, lst2)
                # print(method_name_split)
            elif "backward" in method_name:
                method_name_split = method_name[12:]
                lst2 = generateNew_Method_Name(encoding_types=[method_name_split,])
                lst1 =del_lst2_from_lst1(lst1, lst2)
                # print(method_name_split)
            else:
                print("Error")


    method = ["feature-DT.csv", "-PDT-Profile.csv", "-Top-n-gram.csv", "-PSSM-DT.csv", "-CC-PSSM.csv", "-AC-PSSM.csv",
              "ACC-PSSM.csv", "kmer", "feature-AC.csv", "ACC.csv", "feature-CC.csv", "DP.csv", "DR.csv",
              "PC-PseAAC.csv", "PC-PseAAC-General.csv", "PDT.csv", "SC-PseAAC.csv", "SC-PseAAC-General.csv"]
    method = method + new_method_list
    datadics = datadic(filegroup, method)
    datadics_list = generate_data_group(datadics)
    feature_num = len(method)
    #result = train_GBM_data_group(datadics_list, feature_num)
    result = train_Catboost_data_group(datadics_list, feature_num)
    print(result)
if __name__ == '__main__':
    main()
