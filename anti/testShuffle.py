import pandas as pd
import random
import numpy as np
def split_data(data, k, random_num):
    data = np.array(data)
    x = [i for i in range(260)]
    # random.Random(random_num).shuffle(x)
    # print(x.index(0))
    # print(data.shape)
    data_num = data.shape[0]
    data_split_data_number = int(data_num / k)
    data_split_list = []

    for i in range(10):
        # print(i * data_split_data_number)
        # print((i + 1) * data_split_data_number)
        data_split_list.append(data[x[i*data_split_data_number:(i+1)*data_split_data_number]])

    return np.array(data_split_list)

# def split_data(data, k, random_num):
#     data = np.array(data)
#     print(data.shape)
#     x = [i for i in range(260)]
#     random.Random(random_num).shuffle(x)
#     print(data.shape)
#     data_num = data.shape[0]
#     data_split_data_number = int(data_num / k)
#     data_split_list = []
#     for i in range(10):
#         print(i*data_split_data_number,(i+1)*data_split_data_number)
#         data_split_list.append(data[i*data_split_data_number:(i+1)*data_split_data_number])
#
#     return np.array(data_split_list)
def Catboost_Rank_file():
    data = pd.read_csv("Catboost_Rank\\Catboost_ranking.csv"
    ,header=0,index_col=0)
    rank_ind = data.index
    return  rank_ind

def Ranking_Acc():

    data = pd.read_csv("Result\\Independent_Metric_First_Layer\\A_ranking.csv"
    ,header=0,index_col=0)
    rank_ind = data.index
    return  rank_ind

#print(data)
# data = random_split_data(data,10,10)




