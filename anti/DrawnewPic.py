#-*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



params = {
    'figure.figsize': '7.08, 8'
}
plt.rcParams.update(params)
plt.rcParams['savefig.dpi'] = 300  #00 图片像素
plt.rcParams['figure.dpi'] = 300  # 分

filepath_cv = "Independent_metric\\Different_method.csv"


Amino_acid_composition=["kmer","DR","DP"]
Auto_correlation=["AC", "CC","ACC", "PDT"]
Pseudo_amino_acid_composition =["PC-PseAAC", "PC-PseAAC-General", "SC-PseAAC", "SC-PseAAC-General"]
Profile_based_features =["DT", "PDT-Profile", "Top-n-gram", "PSSM-RT", "PSSM-DT", "CC-PSSM",
          "AC-PSSM", "ACC-PSSM" ]
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }
font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }

font3 = {'family': 'Times New Roman',
         'weight': 'heavy',
         'size': 24,
         }
df = pd.read_csv(filepath_cv,index_col=0)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(100, 100))
#colorslist = ['gray','aqua','#0343df','lime']
#cmaps = colors.LinearSegmentedColormap.from_list('mylist',colorslist,N=800)
ax1 = df.loc[Amino_acid_composition,:].T.plot(ax=axes[0, 0],mark_right=False,kind="bar",figsize=(15,13),title="Amino_acid_composition",rot=360,ylim=(0.35,1.0));
# ax1.legend(loc=9)
ax1.set_ylabel("Ratio",font2)
ax1.text(-1,1.05,"A",font3)
ax1.set_xlabel("Evaluation metrics",font2)
axes[0, 0].set_title('Amino acid composition',font2)


ax2=df.loc[Auto_correlation,:].T.plot(ax=axes[0, 1],kind="bar",figsize=(15,13),title="Auto_correlation",rot=360,ylim=(0.35,1.0));
ax2.set_ylabel("Ratio",font2)
# ax2.legend(loc=9)
ax2.text(-1,1.05,"B",font3)
ax2.set_xlabel("Evaluation metrics",font2)
axes[0, 1].set_title('Auto correlation',font2)

ax3=df.loc[Pseudo_amino_acid_composition,:].T.plot(ax=axes[1, 0],figsize=(15,13),title="Pseudo_amino_acid_composition",kind="bar",rot=360,ylim=(0.50,1.0));
ax3.set_ylabel("Ratio",font2)
ax3.set_xlabel("Evaluation metrics",font2)
axes[1, 0].set_title('Pseudo amino acid composition',font2)
ax3.text(-1,1.02,"C",font3)
ax4=df.loc[Profile_based_features,:].T.plot(ax=axes[1, 1],kind="bar",figsize=(15,13),title="Profile_based_features",rot=360,ylim=(0.45,1.0));
ax4.set_ylabel("Ratio",font2)
ax4.set_xlabel("Evaluation metrics",font2)
axes[1, 1].set_title('Profile-based features',font2)
ax4.text(-1,1.02,"D",font3)
# plt.savefig("picture\\different_method_4-18")
# plt.axis('off')
fig.savefig("picture\\different_method_4-18",dpi=300)