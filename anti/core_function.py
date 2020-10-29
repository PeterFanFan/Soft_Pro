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

encoding_types = ['One_hot', 'One_hot_6_bit', 'Binary_5_bit', 'Hydrophobicity_matrix',
                  'Meiler_parameters', 'Acthely_factors', 'PAM250', 'BLOSUM62', 'Miyazawa_energies',
                  'Micheletti_potentials', 'AESNN3', 'ANN4D', 'ProtVec']
# 将数据分为十组
def model_Predict(data):
    from cls import  sssvm
    from old_anti_cv_gbm import svm_best_parameters_cross_validation
    # model = svm_best_parameters_cross_validation(data[0], data[1])
    # indemetric = sssvm(data[2], data[3], model)
    #indemetric = svm_opt(data[0],data[1],data[2], data[3])
    indemetric = svmOpt(data[0],data[1],data[2], data[3])
    random_inde =randomforest(data[0],data[1],data[2], data[3])
    bayes_inde =bayes(data[0],data[1],data[2], data[3])
    logisticregression_inde =logisticregression(data[0],data[1],data[2], data[3])
    adaboost_inde =adaboost(data[0],data[1],data[2], data[3])
    knn_inde = knn(data[0],data[1],data[2], data[3])
    decisiontree_inde = decisiontree(data[0],data[1],data[2], data[3])
    catboost_inde =catboost_cl(data[0],data[1],data[2], data[3])
    lightgbm_inde = lightgbm(data[0],data[1],data[2], data[3])

    metric = pd.DataFrame(indemetric)
    random_inde =pd.DataFrame(random_inde)
    bayes_inde =pd.DataFrame(bayes_inde)
    logisticregression_inde =pd.DataFrame(logisticregression_inde)
    adaboost_inde =pd.DataFrame(adaboost_inde)
    knn_inde = pd.DataFrame(knn_inde)
    decisiontree_inde =pd.DataFrame(decisiontree_inde)
    catboost_inde = pd.DataFrame(catboost_inde)
    lightgbm_inde = pd.DataFrame(lightgbm_inde)


    col = ['acc','auc','sen','spec','mcc','f1_score']
    piece = metric.loc[0, col]
    random_inde_piece = random_inde.loc[0, col]
    bayes_inde_piece = bayes_inde.loc[0, col]
    logisticregression_inde_piece = logisticregression_inde.loc[0, col]
    adaboost_inde_piece = adaboost_inde.loc[0, col]
    knn_inde_piece = knn_inde.loc[0, col]
    decisiontree_inde_piece =decisiontree_inde.loc[0, col]
    catboost_inde_piece = catboost_inde.loc[0, col]
    lightgbm_inde_piece = lightgbm_inde.loc[0, col]

    piece.name='Svm'
    random_inde_piece.name='Randomforest'
    bayes_inde_piece.name='Bayes'
    logisticregression_inde_piece.name='logisticregression'
    adaboost_inde_piece.name='Adaboost'
    knn_inde_piece.name = 'Knn'
    decisiontree_inde_piece.name='Decision_tree'
    catboost_inde_piece.name = 'Catboost'
    lightgbm_inde_piece.name = "lightgbm"

    outCome = pd.concat([piece,catboost_inde_piece,knn_inde_piece,logisticregression_inde_piece,random_inde_piece,bayes_inde_piece,adaboost_inde_piece,decisiontree_inde_piece,lightgbm_inde_piece],axis=1)

   #  nowtime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
   #  file = "independent_test_" + nowtime + ".csv"
   #  filepath = "Result\\Independent_Metric_Second_Layer\\" + file
   #  # print("Split_Result")
   # # print(outCome)
   #  outCome.to_csv(filepath)
    return  outCome
#
def train_Catboost_data_group_Rank(datadics_list):
    first_layer_result_list = []
    for i in range(10):
        # print("Split: "+str(i))
        datadics = datadics_list[i]
        data, result = trainmodel_CatBoost_Rank(datadics, i+1)
        first_layer_result_list.append(result)

    aver_outcome = first_layer_result_list[0]
    filepath = "Catboost_Rank/Catboost_Ave_Rank.csv"
    for i in range(1, 10):
        aver_outcome = aver_outcome + first_layer_result_list[i]
    aver_outcome = aver_outcome / 10
    aver_outcome = aver_outcome.sort_values(by="Acc", ascending=False)
    aver_outcome.to_csv(filepath)

def train_Lightgbm_data_group_Rank(datadics_list, data_name):
    first_layer_result_list = []
    for i in range(10):
        # print("Split: "+str(i))
        datadics = datadics_list[i]
        data, result = trainmodel_Lightgbm_Rank(datadics, i + 1)
        first_layer_result_list.append(result)

    aver_outcome = first_layer_result_list[0]
    filepath = "Lightgbm_Rank_/Catboost_Ave_Rank.csv"
    for i in range(1, 10):
        aver_outcome = aver_outcome + first_layer_result_list[i]
    aver_outcome = aver_outcome / 10
    aver_outcome = aver_outcome.sort_values(by="Acc", ascending=False)
    aver_outcome.to_csv(filepath)

        # outcome = model_Predict(data)
        # outcom_list.append(outcome)
def train_Catboost_data_group(datadics_list, feature_num):
    outcom_list = []
    datadics_list_depp_copy = copy.deepcopy(datadics_list)
    for i in range(10):
        # print("Split: "+str(i))

        # print("原始数据集")
        datadics = datadics_list_depp_copy[i]
        datadics = copy.deepcopy(datadics)
        for j in datadics:
            data = datadics[j]
            # print("Train data shape", i, data[0].shape)
            # print("Test data shape", i, data[2].shape)
        data = trainmodel_CatBoost(datadics)
        outcome = model_Predict(data)
        outcom_list.append(outcome)

    aver_outcome = outcom_list[0]
    for i in range(1,10):
        aver_outcome = aver_outcome + outcom_list[i]
    file = "Aver_Result_Rank_Catboost"+str(feature_num)+".csv"
    filepath = "Result\\Independent_Metric_Second_Layer\\" + file
    aver_outcome = aver_outcome / 10
    # print("file_name")
    aver_outcome.to_csv(filepath)
    return aver_outcome
import copy
def train_GBM_data_group(datadics_list, feature_num):
    outcom_list = []
    datadics_list_depp_copy = copy.deepcopy(datadics_list)

    for i in range(10):
        # print("Split: "+str(i))
        datadics = datadics_list_depp_copy[i]
        datadics = copy.deepcopy(datadics)
        #datadics = datadics_list[i]
        data = trainmodel_GBM(datadics)
        outcome = model_Predict(data)
        outcom_list.append(outcome)

    aver_outcome = outcom_list[0]
    for i in range(1,10):
        aver_outcome = aver_outcome + outcom_list[i]
    file = "Aver_Result_Lightgbm_"+str(feature_num)+".csv"
    filepath = "Result\\Independent_Metric_Second_Layer\\" + file
    aver_outcome = aver_outcome / 10
    # print("file_name")
    aver_outcome.to_csv(filepath)
    return aver_outcome
#
def generate_data_group(datadics):
    datadics_list = []
    tmp_datadic = {}

    for i in range(10):
        for method in datadics:
            # print(datadics[method][0].shape)
            # print(datadics[method][3][0])
            tmp_datadic[method] = [datadics[method][0],datadics[method][1],datadics[method][2][i],datadics[method][3][i]]
        datadics_list.append(tmp_datadic)
    print("生成数据")
    # for i in range(10):
    #     print("生成数据集: "+str(i))
    #
    #     datadics = datadics_list[i]
    #     for i in datadics:
    #         data = datadics[i]
    #         print("Train data shape", i, data[0].shape)
    #         print("Test data shape", i, data[2].shape)
    return datadics_list
# outCome.plot()

def dataprocessingCV(filepath, methodname, k=0):
    # print ("Loading feature files")
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
    under_model = EasyEnsemble(random_state=46, n_subsets=10)
    traindata = pd.concat([dataset2, dataset4], axis=0)
    negtraintags = [0] * dataset2.shape[0]
    postraintags = [1] * dataset4.shape[0]
    traintags = negtraintags + postraintags
    # traindata, traintags = under_model.fit_sample(traindata, traintags)
    traindata, traintags = under_model.fit_sample(traindata, traintags)

    # print(len(traintags))
    data = [traindata, traintags]
    return data

def dataprocessing(filepath, methodname):
    # print ("Loading feature files")
    if methodname in generateNew_Method_Name():

        dataset1 = pd.read_csv(filepath[0], header=0, index_col=0, low_memory=False)
        dataset2 = pd.read_csv(filepath[1], header=0, index_col=0, low_memory=False)
        dataset3 = pd.read_csv(filepath[2], header=0, index_col=0, low_memory=False)
        dataset4 = pd.read_csv(filepath[3], header=0, index_col=0, low_memory=False)
    else:

    #print (filepath)
        dataset1 = pd.read_csv(filepath[0], header=None, low_memory=False)
        dataset2 = pd.read_csv(filepath[1], header=None, low_memory=False)
        dataset3 = pd.read_csv(filepath[2], header=None, low_memory=False)
        dataset4 = pd.read_csv(filepath[3], header=None, low_memory=False)
    # dataset1 = pd.read_csv('neg-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset2 = pd.read_csv('neg-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset3 = pd.read_csv('pos-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset4 = pd.read_csv('pos-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset1=pd.DataFrame(dataset1,dtype=np.float)
    # print ("Feature processing")
    dataset1 = dataset1.convert_objects(convert_numeric=True)
    dataset2 = dataset2.convert_objects(convert_numeric=True)
    dataset3 = dataset3.convert_objects(convert_numeric=True)
    dataset4 = dataset4.convert_objects(convert_numeric=True)
    dataset1.dropna(inplace=True)
    dataset2.dropna(inplace=True)
    dataset3.dropna(inplace=True)
    dataset4.dropna(inplace=True)

    # 将负样本分割成10份
    dataset1_list = []
    # print(dataset1)
    dataset1 = split_data(dataset1, 10, random_num=10)


    traindata = pd.concat([dataset2, dataset4], axis=0)
    testdata_list = []
    for i in range(10):
        tmp_dataset1 = dataset1[i]
        # print(tmp_dataset1.shape[1])

        tmp_dataset1 = pd.DataFrame(tmp_dataset1,columns=dataset3.columns)
            #print(pd.concat([tmp_dataset1, dataset3], axis=0))
        testdata_split =pd.concat([tmp_dataset1, dataset3], axis=0)
        testdata_split.index = range(testdata_split.shape[0])
        testdata_list.append(testdata_split)

        # print(pd.concat([tmp_dataset1, dataset3], axis=0))
    # testdata = pd.concat([dataset1, dataset3], axis=0)

    #smo = SMOTE(random_state=42)
    from imblearn.under_sampling import RandomUnderSampler
    # 建立模型
    under_model = RandomUnderSampler(random_state=42)
    # 欠处理

    negtraintags = [0] * dataset2.shape[0]
    postraintags = [1] * dataset4.shape[0]
    traintags = negtraintags + postraintags
    # testdata = pd.concat([dataset1, dataset3])


    testtags_list    = []
    postesttags = [1] * dataset3.shape[0]
    for i in range(10):
        tmp_negtesttags = [0] * dataset1[i].shape[0]
        testtags_list.append(tmp_negtesttags+postesttags)

        # negtesttags = [0] * dataset1.shape[0]
    # negtesttags = [0] * dataset1.shape[0]
    # testtags = negtesttags + postesttags
    #traindata, traintags = smo.fit_sample(traindata, traintags)
    traindata, traintags = under_model.fit_sample(traindata, traintags)
    # data = [traindata, traintags, testdata, testtags]
    # print(traindata.shape[1])
    # print(testdata_list[0].shape[1])
    data = [traindata, traintags, testdata_list, testtags_list]
    return data
# 得到路径下所有数据编号
def generateNew_Method_Name(encoding_types=encoding_types, step=2, min_len=32):

    # encoding_types = ['ProtVec']
    # 'ProtVec'
    new_method_name = []
    # min_len = 32
    begin = int(min_len / 2)
    for encoding_type in encoding_types:
        for i in range(begin, min_len + 1,step):
            forward_methodname = "forward_" + str(i) + "_" + encoding_type
            backward_methodname = "backward_" + str(i) + "_" + encoding_type
            new_method_name.append(forward_methodname)
            new_method_name.append(backward_methodname)
    return new_method_name
#
def matchfiles(tmpdir, suffix):  # 读取文件路径
    ###windows os
    # f = glob.glob(tmpdir + '\\*.' + suffix)
    ###linux os
    fi = []
    filenames = []
    # f = glob.glob(tmpdir + suffix)
    f = glob.glob(tmpdir + '/*.' + suffix)
    return f
# 合并编号对应阳阴训练集或测试集并保存
def datadicCV(filegroup, method):
    # method = ["-PSSM-RT.csv","-PSSM-DT.csv","CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","fastafeature-AC.csv","ACC.csv","fastafeature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    # method = ["-PSSM-DT.csv"]
    # method = ["-DT.csv", "-PDT-Profile.csv", "-Top-n-gram.csv", "-PSSM-RT.csv", "-PSSM-DT.csv", "-CC-PSSM.csv",
    #           "-AC-PSSM.csv", "ACC-PSSM.csv", "kmer", "feature-AC.csv", "ACC.csv", "feature-CC.csv", "DP.csv", "DR.csv",
    #           "PC-PseAAC.csv", "PC-PseAAC-General.csv", "PDT.csv", "SC-PseAAC.csv", "SC-PseAAC-General.csv"]
    # method = ["-DT.csv","-PDT-Profile.csv","-Top-n-gram.csv","-CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]

    # method = ["-PSSM-RT.csv","-PSSM-DT.csv","CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","fastafeature-AC.csv","ACC.csv","fastafeature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    # method = ["kmer","fastafeature-AC.csv"]

    postrain = filegroup["postrain"]
    negtrain = filegroup["negtrain"]
    file_method = {}
    filepath = []
    for methodname in method:
        for i in postrain:
            if methodname in i:
                postrain_method = i
                break
        # 匹配出methodname对应的文件
        for j in negtrain:
            if methodname in j:
                negtrain_method = j
                break
        # 匹配出methodname对应的文件

        # dataset1 = pd.read_csv('neg-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
        # dataset2 = pd.read_csv('neg-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
        # dataset3 = pd.read_csv('pos-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
        # dataset4 = pd.read_csv('pos-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
        filepath = [negtrain_method, postrain_method]
        file_method[methodname] = dataprocessingCV(filepath, methodname)
        filepath = []
    return file_method

def dataprocessing_nomean(filepath,methodname="no_name"):

    # print ("Loading feature files")
    print(methodname)
    if methodname in  generateNew_Method_Name(step=2, min_len=6):
        print ("here")

        dataset1 = pd.read_csv(filepath[0], header=0, index_col=0, low_memory=False)
        dataset2 = pd.read_csv(filepath[1], header=0, index_col=0, low_memory=False)
        dataset3 = pd.read_csv(filepath[2], header=0, index_col=0, low_memory=False)
        dataset4 = pd.read_csv(filepath[3], header=0, index_col=0, low_memory=False)
    else:

    #print (filepath)
        dataset1 = pd.read_csv(filepath[0], header=None, low_memory=False)
        dataset2 = pd.read_csv(filepath[1], header=None, low_memory=False)
        dataset3 = pd.read_csv(filepath[2], header=None, low_memory=False)
        dataset4 = pd.read_csv(filepath[3], header=None, low_memory=False)
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
    dataset1.dropna(inplace=True)
    dataset2.dropna(inplace=True)
    dataset3.dropna(inplace=True)
    dataset4.dropna(inplace=True)

    traindata = pd.concat([dataset2, dataset4], axis=0)
    testdata = pd.concat([dataset1, dataset3], axis=0)

    #smo = SMOTE(random_state=42)

    negtraintags = [0] * dataset2.shape[0]
    postraintags = [1] * dataset4.shape[0]
    traintags = negtraintags + postraintags
    # testdata = pd.concat([dataset1, dataset3])
    negtesttags = [0] * dataset1.shape[0]
    postesttags = [1] * dataset3.shape[0]
    testtags = negtesttags + postesttags
    #traindata, traintags = smo.fit_sample(traindata, traintags)
    data = [traindata, traintags, testdata, testtags]
    return data

# 合并编号对应阳阴训练集或测试集并保存
def datadic_nomean(filegroup, method):
    # 11 features
    #    method = ["kmer","feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    #    method = ["feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]

    #    method = ["ACC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    #    method = ["feature-AC.csv","DP.csv"]
    #    method = ["-PSSM-RT.csv","-PSSM-DT.csv","-CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    # method = ["-DT.csv", "-PDT-Profile.csv", "-Top-n-gram.csv", "-PSSM-RT.csv", "-PSSM-DT.csv", "-CC-PSSM.csv",
    #           "-AC-PSSM.csv", "ACC-PSSM.csv", "kmer", "feature-AC.csv", "ACC.csv", "feature-CC.csv", "DP.csv", "DR.csv",
    #           "PC-PseAAC.csv", "PC-PseAAC-General.csv", "PDT.csv", "SC-PseAAC.csv", "SC-PseAAC-General.csv"]
    # method = ["-DT.csv", "-PDT-Profile.csv", "-Top-n-gram.csv", "-PSSM-DT.csv", "-CC-PSSM.csv", "-AC-PSSM.csv",
    #           "ACC-PSSM.csv", "kmer", "feature-AC.csv", "ACC.csv", "feature-CC.csv", "DP.csv", "DR.csv",
    #           "PC-PseAAC.csv", "PC-PseAAC-General.csv", "PDT.csv", "SC-PseAAC.csv", "SC-PseAAC-General.csv"]
    # #
    # new_method_name = generateNew_Method_Name()
    # new_method_name = []
    # method = method + new_method_name
    # method = ['AC.csv']

    postrain = filegroup["postrain"]
    negtrain = filegroup["negtrain"]
    postest = filegroup["postest"]
    negtest = filegroup["negtest"]
    file_method = {}
    filepath = []
    for methodname in method:
        for i in postrain:
            if methodname in i:
                postrain_method = i
                break
        # 匹配出methodname对应的文件
        for j in negtrain:
            if methodname in j:
                negtrain_method = j
                break
        # 匹配出methodname对应的文件
        for k in postest:
            if methodname in k:
                postest_method = k
                break
        # 匹配出methodname对应的文件
        for l in negtest:
            if methodname in l:
                negtest_method = l
                break
        # dataset1 = pd.read_csv('neg-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
        # dataset2 = pd.read_csv('neg-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
        # dataset3 = pd.read_csv('pos-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
        # dataset4 = pd.read_csv('pos-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
        filepath = [negtest_method, negtrain_method, postest_method, postrain_method]
        # print (filepath)
        file_method[methodname] = dataprocessing_nomean(filepath, methodname)
    #print file_method
    return file_method
def datadic(filegroup, method):
    postrain = filegroup["postrain"]
    negtrain = filegroup["negtrain"]
    postest = filegroup["postest"]
    negtest = filegroup["negtest"]
    file_method = {}
    filepath = []
    for methodname in method:
        # print(methodname)
        for i in postrain:
            if methodname in i:
                postrain_method = i
                break
        # 匹配出methodname对应的文件
        for j in negtrain:
            if methodname in j:
                negtrain_method = j
                break
        # 匹配出methodname对应的文件
        for k in postest:
            if methodname in k:
                postest_method = k
                break
        # 匹配出methodname对应的文件
        for l in negtest:
            if methodname in l:
                negtest_method = l
                break
        # dataset1 = pd.read_csv('neg-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
        # dataset2 = pd.read_csv('neg-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
        # dataset3 = pd.read_csv('pos-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
        # dataset4 = pd.read_csv('pos-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
        filepath = [negtest_method, negtrain_method, postest_method, postrain_method]
        # print (filepath)
        file_method[methodname] = dataprocessing(filepath, methodname)



    # print file_method
    return file_method
# get best svm parameters
def optimize_svm(train_x, train_y):
    k = 5
    KF = KFold(n_splits=k, shuffle=True, random_state=5)
    c = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000]
    gamma = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
    aucDic = {}
    accDic = {}
    selected_train = train_x
    traintags = train_y
    for i in c:
        for j in gamma:
            aucList = []
            accList = []
            for train_index, test_index in KF.split(range(0, len(train_y))):
                X_train, X_test = np.array(selected_train)[train_index], np.array(selected_train)[test_index]
                Y_train, Y_test = np.array(traintags)[train_index], np.array(traintags)[test_index]
                from sklearn.svm import SVC
                model = SVC(C=i, gamma=j, probability=True)
                model.fit(X_train, Y_train)
                y_score = model.predict_proba(X_test)
                y_pred = model.predict(X_test)
                acc_item = accuracy_score(Y_test, y_pred, normalize=True)
                auc_item = roc_auc_score(Y_test, y_score[:, 1])
                aucList.append(auc_item)
                accList.append(acc_item)
            auc_mean = np.mean(aucList)
            acc_mean = np.mean(accList)
            aucDic[auc_mean] = [i, j]
            accDic[acc_mean] = [i, j]
    auc_max = max(aucDic.keys())
    acc_max = max(accDic.keys())
    c, gamma = aucDic[auc_max]
    c, gamma = accDic[acc_max]
    print aucDic
    print "C: " + str(c)
    print "Gamma: " + str(gamma)

    model = SVC(C=c, gamma=gamma, probability=True)
    model.fit(train_x, train_y)
    return model

# 多重子模型分类
def trainmodel_mutipleModel(datadic):
    train_feature = {}
    test_feature = {}
    metrics = {}
    index = []
    for i in datadic:
        data = datadic[i]

        (y_pred_train, y_pred_test) = trainmodel_multipleTags(data[0], data[1], data[2], data[3])
        # y_pred_train_和y_pred_test 是多个预测结果的列表

        feature_SVM = i + "_" + "SVM"
        feature_DT = i + "_" + "DT"
        feature_NB = i + "_" + "NB"
        feature_Knn = i + "_" + "Knn"

        train_feature[feature_Knn] = y_pred_train[0]
        train_feature[feature_NB] = y_pred_train[1]
        train_feature[feature_SVM] = y_pred_train[2]
        train_feature[feature_DT] = y_pred_train[3]

        test_feature[feature_Knn] = y_pred_test[0]
        test_feature[feature_NB] = y_pred_test[1]
        test_feature[feature_SVM] = y_pred_test[2]
        test_feature[feature_DT] = y_pred_test[3]

    train_feature_vector = pd.DataFrame(train_feature)
    test_feature_vector = pd.DataFrame(test_feature)
    data[0] = train_feature_vector.values
    data[2] = test_feature_vector.values
    return data

# 子模型分类
def trainmodel_CatBoost_Rank(datadic, m):
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
        # print ("This is training label")
        # print i

        data = datadic[i]
        # print(data[0].head())
        # print(data[2].head())
        # print(data[0].shape[1])
        # print(data[2].shape[1])
        index.append(i)
        #        params = opt(data[0],data[1])

        #        indemetric = lightgbm(data[0],data[1],data[2],data[3],params)
        #        y_pred_train= model.predict(data[0])
        #        y_pred_test = model.predict(data[2])

        (y_pred_train, y_pred_test, metric) = catboost_TrainStag(data[0],data[1],data[2],data[3])
        metrics['Mcc'].append(metric['mcc'])
        metrics['Acc'].append(metric['acc'])
        metrics['Sen'].append(metric['sen'])
        metrics['Sp'].append(metric['spec'])
        metrics['Auc'].append(metric['auc'])
        train_feature[i] = y_pred_train
        test_feature[i] = y_pred_test
        # data = [traindata,traintags,testdata,testtags]

    outcome = pd.DataFrame(metrics, index=index)
    file = "Catboost_Rank_" +str(m)+".csv"
    filepath = "Catboost_Rank\\"+file
    if (os.path.exists(filepath)):
        result = pd.read_csv(filepath, index_col=0, header=0)
    else:
        outcome.to_csv(filepath)
    train_feature_vector = pd.DataFrame(train_feature)
    test_feature_vector = pd.DataFrame(test_feature)
    data[0] = train_feature_vector.values
    data[2] = test_feature_vector.values
    #    data[0] = train_feature_vector
    #    data[2] = test_feature_vector
    return data, outcome
def trainmodel_Lightgbm_Rank(datadic, m):
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
        # print ("This is training label")
        # print i

        data = datadic[i]
        # print(data[0].head())
        # print(data[2].head())
        # print(data[0].shape[1])
        # print(data[2].shape[1])
        index.append(i)
        #        params = opt(data[0],data[1])

        #        indemetric = lightgbm(data[0],data[1],data[2],data[3],params)
        #        y_pred_train= model.predict(data[0])
        #        y_pred_test = model.predict(data[2])

        #(y_pred_train, y_pred_test, metric) = catboost_TrainStag(data[0],data[1],data[2],data[3])
        (y_pred_train, y_pred_test, metric) = lightgbmTrainStag(data[0],data[1],data[2],data[3])
        metrics['Mcc'].append(metric['mcc'])
        metrics['Acc'].append(metric['acc'])
        metrics['Sen'].append(metric['sen'])
        metrics['Sp'].append(metric['spec'])
        metrics['Auc'].append(metric['auc'])
        train_feature[i] = y_pred_train
        test_feature[i] = y_pred_test
        # data = [traindata,traintags,testdata,testtags]

    outcome = pd.DataFrame(metrics, index=index)
    file = "Lightgbm_Rank_" +str(m)+".csv"
    filepath = "Catboost_Rank\\"+file
    if (os.path.exists(filepath)):
        result = pd.read_csv(filepath, index_col=0, header=0)
    else:
        outcome.to_csv(filepath)
    train_feature_vector = pd.DataFrame(train_feature)
    test_feature_vector = pd.DataFrame(test_feature)
    data[0] = train_feature_vector.values
    data[2] = test_feature_vector.values
    #    data[0] = train_feature_vector
    #    data[2] = test_feature_vector
    return data, outcome
def trainmodel_CatBoost(datadic, CV=False, data_num="no_num",  split="no_split"):

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
        # print ("This is training label")
        # print i

        data = datadic[i]
        # print(data[0].head())
        # print(data[2].head())
        # print(data[0].shape[1])
        # print(data[2].shape[1])
        index.append(i)
        #        params = opt(data[0],data[1])

        #        indemetric = lightgbm(data[0],data[1],data[2],data[3],params)
        #        y_pred_train= model.predict(data[0])
        #        y_pred_test = model.predict(data[2])

        # print("Train data shape",i,data[0].shape)
        # print("Test data shape",i,data[2].shape)
        (y_pred_train, y_pred_test, metric) = catboost_TrainStag(data[0],data[1],data[2],data[3],feature_name=i, CV=CV, data_num=data_num,  split=split)
        metrics['Mcc'].append(metric['mcc'])
        metrics['Acc'].append(metric['acc'])
        metrics['Sen'].append(metric['sen'])
        metrics['Sp'].append(metric['spec'])
        metrics['Auc'].append(metric['auc'])
        train_feature[i] = y_pred_train
        test_feature[i] = y_pred_test
        # data = [traindata,traintags,testdata,testtags]
    import time
    outcome = pd.DataFrame(metrics, index=index).sort_values(by="Acc", ascending=False)
    nowtime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    file = "independent_test_" + nowtime + ".csv"
    filepath = "Result\\Independent_Metric_First_Layer\\" + file
    outcome.to_csv(filepath)
    train_feature_vector = pd.DataFrame(train_feature)
    test_feature_vector = pd.DataFrame(test_feature)
    data[0] = train_feature_vector.values
    data[2] = test_feature_vector.values
    #    data[0] = train_feature_vector
    #    data[2] = test_feature_vector
    return data
import joblib
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
        # print ("This is training label")
        # print i
        print(i)
        data = datadic[i]
        # print(data[0].head())
        # print(data[2].head())
        # print(data[0].shape[1])
        # print(data[2].shape[1])
        index.append(i)
        cls = LGBMClassifier()
        params = cls.get_params()
        #        params = opt(data[0],data[1])

        #        indemetric = lightgbm(data[0],data[1],data[2],data[3],params)
        #        y_pred_train= model.predict(data[0])
        #        y_pred_test = model.predict(data[2])
        from cls import lightgbmTrainStag_old
        (y_pred_train, y_pred_test, metric) = lightgbmTrainStag(data[0], data[1], data[2], data[3], params,method_name=i)
        #(y_pred_train, y_pred_test) = lightgbmTrainStag_old(data[0], data[1], data[2], data[3], params)
        metrics['Mcc'].append(metric['mcc'])
        metrics['Acc'].append(metric['acc'])
        metrics['Sen'].append(metric['sen'])
        metrics['Sp'].append(metric['spec'])
        metrics['Auc'].append(metric['auc'])
        train_feature[i] = y_pred_train
        test_feature[i] = y_pred_test
        # data = [traindata,traintags,testdata,testtags]
    import time
    outcome = pd.DataFrame(metrics, index=index).sort_values(by="Acc", ascending=False)
    nowtime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    file = "independent_test_" + nowtime + ".csv"
    filepath = "Result\\Independent_Metric_First_Layer\\" + file
    outcome.to_csv(filepath)

    train_feature_vector = pd.DataFrame(train_feature)
    test_feature_vector = pd.DataFrame(test_feature)
    data[0] = train_feature_vector.values
    data[2] = test_feature_vector.values
    #    data[0] = train_feature_vector
    #    data[2] = test_feature_vector
    return data
# 从A列表中删除B列表中的元素
def del_lst2_from_lst1(lst1, lst2):
    return [x for x in lst1 if x not in lst2]
def Catboost_Rank_file(filepath):
    data = pd.read_csv(filepath,header=0,index_col=0)
    rank_ind = data.index
    return  rank_ind
def matchfiles(tmpdir, suffix):  # 读取文件路径
    ###windows os
    # f = glob.glob(tmpdir + '\\*.' + suffix)
    ###linux os
    fi = []
    filenames = []
    # f = glob.glob(tmpdir + suffix)
    f = glob.glob(tmpdir + '/*.' + suffix)
    return f

import os



def Acr_CV(datadics, data_num):
    k = 5
    m = 31
    cvdatadics = {}
    outcom_list = []
    KF = KFold(n_splits=k, shuffle=True, random_state=10)
    count = 1
    for train_index, test_index in KF.split(range(0, 196)):
        print("This CV : ", count)

        for methodname in datadics.keys():
            data = datadics[methodname]
            selected_features = data[0]
            traintags = data[1]
            X_train, X_test = np.array(selected_features)[train_index], np.array(selected_features)[test_index]
            Y_train, Y_test = np.array(traintags)[train_index], np.array(traintags)[test_index]
            newdata = [X_train, Y_train, X_test, Y_test]
            cvdatadics[methodname] = newdata
        ##svm opt
        #        newfeature = trainmodel(cvdatadics)
        ##lightGBM opt
        newfeature = trainmodel_CatBoost(cvdatadics, CV=True, data_num=data_num,  split=count)
        outcome = model_Predict(newfeature)
        print(outcome)
        outcom_list.append(outcome)
        count = count + 1
    aver_outcome = outcom_list[0]
    for i in range(1,k):
        aver_outcome = aver_outcome + outcom_list[i]

    aver_outcome = aver_outcome / k

    return  aver_outcome
def mkd(x):
    if (os.path.exists(x)):
        pass
    else:
        os.makedirs(x)
def Ranking_Acc():

    data = pd.read_csv("Result\\Independent_Metric_First_Layer\\A_ranking.csv"
    ,header=0,index_col=0)
    rank_ind = data.index
    return  rank_ind