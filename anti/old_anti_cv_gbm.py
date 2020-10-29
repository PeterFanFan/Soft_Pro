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

from cls import  lightgbm
def main():
    warnings.filterwarnings("ignore")
    print("hello world")
    tmpdir = "F:\\Programs\\journals_papers\\Anticancer Peptides\\anti-2020-3-27\\data\\AntiCAP_138pos_206neg_2014.321\\method-feature"
    #tmpdir = "E:\\data\ACP_2018_CV\\method-feature"

    suffix  = "csv"
    postrain = glob.glob(tmpdir + '/*138*')
    negtrain = glob.glob(tmpdir + '/*206*')
    # postrain = glob.glob(tmpdir + '/pos*')
    # negtrain = glob.glob(tmpdir + '/neg*')
    print len(postrain)
    filegroup = {}
    filegroup['postrain']= postrain
    filegroup['negtrain']= negtrain

    datadics = datadic(filegroup)

    data = trainmodel(datadics)
    #data = no_opt_trainmodel(datadics)
    
    cls = LGBMClassifier()
    #params = cls.get_params()
    params = opt(data[0],data[1])
    meandic = cvt(data[0], data[1], params)
    metric = pd.DataFrame(meandic)
    col = ['acc','auc','sen','spec','mcc','f1_score']
    print metric.loc[:,col]
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

    #testsdata = pd.concat([dataset1, dataset3])
    #negtesttags = [0]*dataset1.shape[0]
    #postesttags= [1]*dataset3.shape[0]
    #testtags = negtesttags+postesttags
    # 打乱数据集
    # cc = list(zip(traindata.values, traintags))
    # random.shuffle(cc)
    # traindata, traintags = zip(*cc)
    # #print type(traindata)
    # #print traindata.shape
    # traindata, traintags =pd.DataFrame(list(traindata)),list(traintags)
    # print traindata
    # print traintags
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


#得到路径下所有数据编号
def getnamenum(tmpdir,suffix):
    numlist=[]
    f = glob.glob(tmpdir + '/*.' + suffix)
    postrain = glob.glob(tmpdir + '/pos-train*.' + suffix)
    negtrain = glob.glob(tmpdir + '/neg-train*.' + suffix)
    postest  = glob.glob(tmpdir + '/pos-test*.' + suffix)
    negtest  = glob.glob(tmpdir + '/neg-test*.' + suffix)
    for i in postrain:
        fname,filename = os.path.split(i)
        filename = filename.split(".fasta.csv")
        filename = filename[0][10:]
        numlist.append(filename)
    return numlist
    # print (len(postrain))
    # print (len(negtrain))
    # print (len(postrain))
    # print (len(negtrain))
    # print (postrain)
    # print (negtrain)
    # print (postrain)
    # print (negtrain)
def mkd(x):
    if(os.path.exists(x)):
        pass
    else:
        os.makedirs(x)

#输入文件名编号，找出4个文件的44个特征文件路径
def matchnum(filenum,tmpdir,suffix):
	#对应编号的一组文件
    filegroup = {}
    postrain = glob.glob(tmpdir + '/pos-train*'+filenum+'*' + suffix)
    negtrain = glob.glob(tmpdir + '/neg-train*'+filenum+'*' + suffix)
    postest = glob.glob(tmpdir + '/pos-test*'+filenum+'*' + suffix)
    negtest = glob.glob(tmpdir + '/neg-test*'+filenum+'*' + suffix)
    filegroup["postrain"] = postrain
    filegroup["negtrain"] = negtrain
    filegroup["postest"] = postest
    filegroup["negtest"] = negtest
    return filegroup

#合并编号对应阳阴训练集或测试集并保存
def datadic(filegroup):
    #method = ["-PSSM-RT.csv","-PSSM-DT.csv","CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","fastafeature-AC.csv","ACC.csv","fastafeature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    #method = ["-PSSM-DT.csv"]
    #method = ["PDT-Profile.csv","PSSM-DT.csv","PSSM-RT.csv","DT.csv","-CC-PSSM.csv","AC-PSSM.csv","ACC-PSSM.csv","kmer","fastafeature-AC.csv","ACC.csv","fastafeature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    method = ["-DT.csv","-PDT-Profile.csv","-Top-n-gram.csv","-CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","feature-AC.csv","ACC.csv","feature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]

    #method = ["-PSSM-RT.csv","-PSSM-DT.csv","CC-PSSM.csv","-AC-PSSM.csv","ACC-PSSM.csv","kmer","fastafeature-AC.csv","ACC.csv","fastafeature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]
    #method = ["kmer","fastafeature-AC.csv","ACC.csv","fastafeature-CC.csv","DP.csv","DR.csv","PC-PseAAC.csv","PC-PseAAC-General.csv","PDT.csv","SC-PseAAC.csv","SC-PseAAC-General.csv"]

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

#get best svm parameters 
def svm_best_parameters_cross_validation(train_x, train_y):    
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000,10000,100000], 'gamma': [1e-5, 1e5]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(train_x, train_y)    
    return model

#训练11个模型
def trainmodel(datadic):
    train_feature = {}
    test_feature  = {}
    for i in datadic:
        data = datadic[i]    
        model=svm_best_parameters_cross_validation(data[0], data[1])
        y_pred_train = model.predict(data[0])
        train_feature[i]=y_pred_train
        
        #data = [traindata,traintags,testdata,testtags]
    train_feature_vector = pd.DataFrame(train_feature)
    # test_feature_vector  = pd.DataFrame(test_feature)
    data[0] = train_feature_vector.values
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
        model.fit(data[0], data[1])
        y_pred_train = model.predict(data[0])
        # y_pred_test = model.predict(data[2])
        print (i)
        train_feature[i]=y_pred_train
        # test_feature[i]=y_pred_test
        #data = [traindata,traintags,testdata,testtags]
    train_feature_vector = pd.DataFrame(train_feature)
    # test_feature_vector  = pd.DataFrame(test_feature)
    data[0] = train_feature_vector.values
    # data[2] = test_feature_vector.values
    return data
	# return model



if __name__ == '__main__':
	main()