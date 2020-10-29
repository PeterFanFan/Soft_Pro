from core_function import *
import copy
import pandas as pd
import os
warnings.filterwarnings("ignore")
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
filepath = "Catboost_Rank/Catboost_Ave_Rank.csv"
method = Catboost_Rank_file(filepath)

filepath = "Result\\CV\\Second_layer"
mkd(filepath)
m=10
f_n=10
for feature_num in range(4, f_n):
    print("Feature num: ", feature_num)
    file = filepath + "\\Aver_CV_Result_Rank_Catboost" + str(feature_num) + ".csv"
    if (os.path.exists(file)):
        Aver_Result = pd.read_csv(file, index_col=0, header=0)
    else:
        # tmp_datadic = {}
        Result_list = []
        method = method[0:feature_num]
        datadics = datadicCV(filegroup, method=method)
        for i in range(m):
            print("This is NO: "+str(i+1)+" data")
            tmp_datadic = {}
            for method in datadics:
                tmp_datadic[method] = copy.deepcopy([datadics[method][0][i], datadics[method][1][i]])
            Result = Acr_CV(tmp_datadic, i+1)
            Result_list.append(Result)
        Aver_Result = Result_list[0]
        for i in range(1,m):
            Aver_Result = Aver_Result + Result_list[i]
        Aver_Result = Aver_Result / m
        Aver_Result.to_csv(file)
    print("Aver_Result: ")
    print(Aver_Result)
