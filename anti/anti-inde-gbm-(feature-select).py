#-*- coding: UTF-8 -*-
from bayesHyperTuning import opt
import pandas as pd
from skfeature.function.sparse_learning_based import ls_l21
from sklearn import svm
from sklearn.svm import SVC
from skfeature.utility.sparse_learning import construct_label_matrix_pan, feature_ranking
import glob
import pandas as pd
from testmetric import *
from lightgbm import  LGBMClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
#from bayesHyperTuning import opt
from cls import randomforest,bayes,logisticregression,decisiontree,adaboost,knn
from cls import  lightgbm
from cls import  lightgbmTrainStag
from cls import  svmOpt

def main():

    #tmpdir = "F:\\Programs\\journals_papers\\Anticancer Peptides\\anti-2020-3-27\\data\\2019-data\\method-feature"
    tmpdir  ="E:\\data\\2019-data\\method-feature"
    #tmpdir  ="E:\\data-AVP\\FASTA\\method-feature-1"
    # tmpdir  ="E:\\data-AVP\\FASTA\\method-feature-2"
    # tmpdir = "F:\\Programs\\journals_papers\\Anticancer Peptides\\anti-2020-3-27\\data\\ACP_2018\\method-feature"
    #tmpdir ="E:\\data\\ACPP_2015supp\\method-feature"
    #tmpdir_ty_inde ="E:\\data\Tyagi_datasets\\method-feature"
    #tmpdir2 = "E:\\data\\AntiCAP_138pos_206neg_2014.321\\method-feature"

    suffix  = "csv"
    postrain = glob.glob(tmpdir + '/bmark-positive80*')
    negtrain = glob.glob(tmpdir + '/bmark-negative80*')
    postest = glob.glob(tmpdir + '/Ind-positive*')
    negtest = glob.glob(tmpdir + '/Ind-negative*')

#    postrain = glob.glob(tmpdir + '/postrain*')
#    negtrain = glob.glob(tmpdir + '/negtrain*')
#    postest = glob.glob(tmpdir + '/postest*')
#    negtest = glob.glob(tmpdir + '/negtest*')
#     postrain = glob.glob(tmpdir + '/Training_AVP544p*')
#     negtrain = glob.glob(tmpdir + '/Train_NonAVP407*')
#     postest = glob.glob(tmpdir + '/valid_AVP60*')
#     negtest = glob.glob(tmpdir + '/valid_nonAVP45*')

    # postrain = glob.glob(tmpdir + '/Training_AVP544p*')
    # negtrain = glob.glob(tmpdir + '/Train_NonAVP407*')
    # postest = glob.glob(tmpdir + '/valid_AVP60*')
    # negtest = glob.glob(tmpdir + '/valid_nonAVP45*')
    #
    # postrain = glob.glob(tmpdir + '/Training_AVP544p*')
    # negtrain = glob.glob(tmpdir + '/AnBP2-NonAVP544-train*')
    # postest = glob.glob(tmpdir + '/valid_AVP60*')
    # negtest = glob.glob(tmpdir + '/AnBP2-NonAVP60-valid*')

    # postrain = glob.glob(tmpdir + '/217_Positive_Train*')
    # negtrain = glob.glob(tmpdir + '/3979_Uniprot_Negative_Test*')
    # print negtrain
    # postest = glob.glob(tmpdir + '/40_Independent_Test_Positive*')
    # negtest = glob.glob(tmpdir + '/40_Independent_Test_Negative*')
    # postest = glob.glob(tmpdir_ty_inde + '/Indep1_50_pos*')
    # negtest = glob.glob(mpdir_ty_inde+ '/Indep1_50_neg*')
    # postest = glob.glob(tmpdir + '/*138*')
    # negtest = glob.glob(tmpdir + '/*206*')

    filegroup = {}
    filegroup['postrain']= postrain
    filegroup['negtrain']= negtrain
    filegroup['postest']= postest
    filegroup['negtest']= negtest

    datadics = datadic(filegroup)
    # print datadics
    data = trainmodel_GBM(datadics)

#   featureselection
    traintagsnump = construct_label_matrix_pan(np.array(data[1]))
    Weight, obj, value_gamma = ls_l21.proximal_gradient_descent(data[0], traintagsnump, 0.1, verbose=False)
    idx = feature_ranking(Weight)
    count = 0
    judge = []
    num = range(1, len(idx), 1)
    pieceList=[]
    for num_fea in num:

        print("Feature number: " + str(num_fea))
        selected_features = data[0][:, idx[0:num_fea]]
        selected_featurestest =data[2][:, idx[0:num_fea]]
        model = SVC(probability=True)
        model.fit(selected_features, data[1])
        indemetric = svmOpt(selected_featurestest, data[3], model)
        metric = pd.DataFrame(indemetric)
        col = ['acc', 'auc', 'sen', 'spec', 'mcc', 'f1_score']
        piece = metric.loc[0, col]
        piece.name=str(num_fea)
        pieceList.append(piece)
    outCome = pd.concat(pieceList,axis=1)
    outCome=outCome.T
    print outCome


    #    cls = LGBMClassifier()
#    #params = cls.get_params()
#    params = opt(data[0],data[1])
#    print params
#    indemetric = lightgbm(data[0],data[1],data[2],data[3],params)
    
    #model = svm_best_parameters_cross_validation(data[0], data[1])
    #indemetric = svmOpt(data[2], data[3], model)
    # #,bayes,logisticregression,decisiontree,adaboost
    # random_inde =randomforest(data[0],data[1],data[2], data[3])
    # bayes_inde =bayes(data[0],data[1],data[2], data[3])
    # logisticregression_inde =logisticregression(data[0],data[1],data[2], data[3])
    # adaboost_inde =adaboost(data[0],data[1],data[2], data[3])
    # knn_inde = knn(data[0],data[1],data[2], data[3])
    # decisiontree_inde = decisiontree(data[0],data[1],data[2], data[3])
    #
    # metric = pd.DataFrame(indemetric)
    # random_inde =pd.DataFrame(random_inde)
    # bayes_inde =pd.DataFrame(bayes_inde)
    # logisticregression_inde =pd.DataFrame(logisticregression_inde)
    # adaboost_inde =pd.DataFrame(adaboost_inde)
    # knn_inde = pd.DataFrame(knn_inde)
    # decisiontree_inde =pd.DataFrame(decisiontree_inde)
    #
    #
    # col = ['acc','auc','sen','spec','mcc','f1_score']
    # piece = metric.loc[0, col]
    # random_inde_piece = random_inde.loc[0, col]
    # bayes_inde_piece = bayes_inde.loc[0, col]
    # logisticregression_inde_piece = logisticregression_inde.loc[0, col]
    # adaboost_inde_piece = adaboost_inde.loc[0, col]
    # knn_inde_piece = knn_inde.loc[0, col]
    # decisiontree_inde_piece =decisiontree_inde.loc[0, col]
    #
    # piece.name='Svm'
    # random_inde_piece.name='Randomforest'
    # bayes_inde_piece.name='Bayes'
    # logisticregression_inde_piece.name='logisticregression'
    # adaboost_inde_piece.name='Adaboost'
    # knn_inde_piece.name = 'Knn'
    # decisiontree_inde_piece.name='Decision_tree'
    #
    # outCome = pd.concat([piece,random_inde_piece,bayes_inde_piece,logisticregression_inde_piece,adaboost_inde_piece,decisiontree_inde_piece],axis=1)
    # # filename="Independent_compare.csv"
    # # filepath = "Independent_metric\\"+filename
    # # outCome.to_csv(filepath)
    # print (outCome)
    # #outCome.plot()

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
#合并编号对应阳阴训练集或测试集并保存
def datadic(filegroup):
    #11 features
#    method = ["kmer","feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
#    method = ["feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]

#    method = ["ACC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
#    method = ["feature-AC.csv","DP.csv"]
#    method = ["-PSSM-RT.csv","-PSSM-DT.csv","-CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    method = ["-DT.csv","-PDT-Profile.csv","-Top-n-gram.csv","-PSSM-RT.csv","-PSSM-DT.csv","-CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    #method = ["-DT.csv","-PDT-Profile.csv","-Top-n-gram.csv","-PSSM-DT.csv","-CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]

    #method = ["-DT.csv","-PDT-Profile.csv","-Top-n-gram.csv","-CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    #method = ["-DT.csv","-PDT-Profile.csv","-Top-n-gram.csv","-PSSM-RT.csv","-PSSM-DT.csv","-CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]

    #method = ['AC.csv']

    postrain = filegroup["postrain"]
    negtrain = filegroup["negtrain"]
    postest  = filegroup["postest"]
    negtest  = filegroup["negtest"]
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
        for k in postest:
            if methodname in k:
                postest_method = k
                break
        #匹配出methodname对应的文件
        for l in negtest:
            if methodname in l:
                negtest_method = l
                break
    # dataset1 = pd.read_csv('neg-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset2 = pd.read_csv('neg-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset3 = pd.read_csv('pos-test.a.3.1.1.fasta.csv',header=None,low_memory=False)
    # dataset4 = pd.read_csv('pos-train.a.3.1.1.fasta.csv',header=None,low_memory=False)
        filepath = [negtest_method,negtrain_method,postest_method,postrain_method]
        print (filepath)
        file_method[methodname] = dataprocessing(filepath)
    return file_method

#get best svm parameters 
def rbf_svm_best_parameters_cross_validation(train_x, train_y):    
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(train_x, train_y)    
    return model
#get best svm parameters 
def svm_best_parameters_cross_validation(train_x, train_y):    
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC    
    model = SVC(probability=False)    
    #param_grid = {'kernel':('linear', 'rbf'),'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}    
    param_grid = {'C': [1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000,10000,100000], 'gamma':[1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000,10000]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = -1, verbose=1,cv=10)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = SVC(C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(train_x, train_y)    
    return model

#训练11个模型
def trainmodel_GBM(datadic):
    train_feature = {}
    test_feature  = {}
    metrics = {}
    index=[]
    metrics['Mcc'] =[]
    metrics['Acc'] =[]
    metrics['Sen'] =[]
    metrics['Sp'] =[]
    metrics['Auc'] =[]
    for i in datadic:
        data = datadic[i]
        index.append(i)
#        model = svm.SVC(probability=False)
#        print("Svm training")
        print i
        print data[0].shape
        print data[2].shape
#        model.fit(data[0], data[1])
        
#        model=svm_best_parameters_cross_validation(data[0], data[1])
        cls = LGBMClassifier()
        params = cls.get_params()
#        params = opt(data[0],data[1])
        
#        indemetric = lightgbm(data[0],data[1],data[2],data[3],params)
#        y_pred_train= model.predict(data[0])
#        y_pred_test = model.predict(data[2])
       
        (y_pred_train,y_pred_test,metric) = lightgbmTrainStag(data[0],data[1],data[2],data[3],params)
        metrics['Mcc'].append(metric['mcc'])
        metrics['Acc'].append(metric['acc'])
        metrics['Sen'].append(metric['sen'])

        metrics['Sp'].append(metric['spec'])
        metrics['Auc'].append(metric['auc'])
        train_feature[i]=y_pred_train
        test_feature[i]=y_pred_test
        #data = [traindata,traintags,testdata,testtags]
    import time
    outcome = pd.DataFrame(metrics,index=index)
    nowtime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    file = "independent_test_"+nowtime+".csv"
    filepath ="Independent_metric\\"+file
    outcome.to_csv(filepath)
    train_feature_vector = pd.DataFrame(train_feature)
    test_feature_vector  = pd.DataFrame(test_feature)
    data[0] = train_feature_vector.values
    data[2] = test_feature_vector.values
#    data[0] = train_feature_vector
#    data[2] = test_feature_vector
    return data
	# return model

#训练11个模型
def no_opt_trainmodel(datadic):
    train_feature = {}
    test_feature  = {}
    for i in datadic:
        data = datadic[i]
        model = svm.SVC(probability=False)
        print("Svm training")
        # print i
        # print data[0].shape
        # print data[2].shape
        model.fit(data[0], data[1])
        y_pred_train = model.predict(data[0])
        y_pred_test = model.predict(data[2])
        train_feature[i]=y_pred_train
        test_feature[i]=y_pred_test
        #data = [traindata,traintags,testdata,testtags]
    # for i in test_feature:
    #     print i
    #     print len(test_feature[i])
    train_feature_vector = pd.DataFrame(train_feature)
    test_feature_vector  = pd.DataFrame(test_feature)
    data[0] = train_feature_vector.values
    data[2] = test_feature_vector.values
#    data[0] = train_feature_vector
#    data[2] = test_feature_vector
    return data
	# return model

if __name__ == '__main__':
	main()