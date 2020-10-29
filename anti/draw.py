#-*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分
# filepath ="anti\\Crossvalidation_metric\\cv_test_2020-04-14-00-28-38.csv"
# data = pd.read_csv(filepath,index_col=0)
# data.plot(kind='box',title="Different partition of data")
#
# plt.savefig("Box.png")
# #plt.show()
# plt.savefig("picture\\Cv_Box.png")

params = {
    'figure.figsize': '7.00, 7.25'
}
plt.rcParams.update(params)
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }
filepath_cv = "Crossvalidation_metric\\Cv_result_compare.csv"
dataCv = pd.read_csv(filepath_cv,index_col=0).sort_index(axis=1)

#ax = dataCv.plot.bar(x="a",y="b",rot=360,ylim=(0.85,1.0))
ax = dataCv.plot(kind="bar",rot=360,ylim=(0.80,1.0))
fig = ax.get_figure()
plt.xlabel("Evaluation metrics",font2)
plt.ylabel("Ratio",font2)
fig.savefig("picture\\Cv_Classify.jpg")




filepath_in = "Independent_metric\\Independent_compare_5_27.csv"
dataIn = pd.read_csv(filepath_in,index_col=0).sort_index(axis=1)

ax = dataIn.plot(kind="bar",rot=360,ylim=(0.55,1.0))
fig_In = ax.get_figure()
plt.xlabel("Evaluation metrics",font2)
plt.ylabel("Ratio",font2)
fig_In.savefig("picture\\In_Classify_5_28.jpg")




filepath_cv_random_state = "Crossvalidation_metric\\cv_test_random_state.csv"
dataCv_random = pd.read_csv(filepath_cv_random_state ,index_col=0).sort_index(axis=1)

ax = dataCv_random.plot(kind="box",rot=360)
fig_Cv_random = ax.get_figure()
plt.xlabel("Evaluation metrics",font2)
plt.ylabel("Ratio",font2)
fig_Cv_random .savefig("picture\\Cv_random_state.jpg")
# filepath_dfmethod = "Independent_metric\Different_method.csv"
# dataDfmethod = pd.read_csv(filepath_dfmethod,index_col=0)
# dataDfmethod = dataDfmethod.sort_index()
# dataDfmethod = dataDfmethod.T

# ax = dataDfmethod.plot(kind="bar",sort_columns=True,title="Different method",rot=360)
# plt.xlabel("12123123")

# fig_dfmethod = ax.get_figure()
# fig_dfmethod.savefig("picture\\Different_method")