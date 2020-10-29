# -*- coding: utf-8 -*-

import numpy as np
import os
import glob
import time
from Bio import SeqIO
# from Bio.Alphabet import IUPAC, ProteinAlphabet
from Bio.Seq import Seq
import re

def count(f):
    countHash_pos =dict(zip("ACDEFGHIKLMNPQRSTVWY",[0]*20))
    countHash_neg =dict(zip("ACDEFGHIKLMNPQRSTVWY",[0]*20))
    countLen_pos ={}
    countLen_neg ={}
    for tmpfile in f:
        if "pos" in tmpfile:
            fasta_file = open(tmpfile, "r")
            for seq_record in SeqIO.parse(fasta_file, "fasta"):
                ## ACDEFGHIKLMNPQRSTVWY
                # print(seq_record.seq)
                seq = seq_record.seq
                for i in seq:
                    countHash_pos[i]+=1
                if(countLen_pos.has_key(len(seq))):
                    countLen_pos[len(seq)]+=1
                else:
                    countLen_pos[len(seq)]=int(1)
        else:
            fasta_file = open(tmpfile, "r")
            for seq_record in SeqIO.parse(fasta_file, "fasta"):
                ## ACDEFGHIKLMNPQRSTVWY
                # print(seq_record.seq)
                seq = seq_record.seq
                for i in seq:
                    countHash_neg[i]+=1
                if(countLen_neg.has_key(len(seq))):
                    countLen_neg[len(seq)]+=1
                else:
                    countLen_neg[len(seq)]=int(1)


            #matchar = re.search(r'[^ACDEFGHIKLMNPQRSTVWY]', str(seq_record.seq), flags=re.I)
    return countHash_pos,countHash_neg,countLen_pos,countLen_neg



filepath1="E:\\data\\2019-data"
filepath2="E:\\data\\138_206"

suffix="fasta"
countHash= dict(zip('ACDEFGHIKLMNPQRSTVWY', [0]*20))

f1 = glob.glob(filepath1 + '/*.' + suffix)
f2 = glob.glob(filepath2 + '/*.' + suffix)
ch_pos1,ch_neg1,cl_pos1,cl_neg1 = count(f1)
ch_pos2,ch_neg2,cl_pos2,cl_neg2 = count(f2)
import pandas as pd
ch_pos1 = pd.Series(ch_pos1)
ch_neg1 = pd.Series(ch_neg1)
cl_pos1 = pd.Series(cl_pos1)
cl_neg1 = pd.Series(cl_neg1)

ch_pos2 = pd.Series(ch_pos2)
ch_neg2 = pd.Series(ch_neg2)
cl_pos2 = pd.Series(cl_pos2)
cl_neg2 = pd.Series(cl_neg2)
# cl1 = pd.Series(cl1)
# cl2 = pd.Series(cl2)


ch_pos2.name="138_206-data-Anti"
ch_neg2.name="138_206-data"

cl_pos2.name="138_206-data-Anti"
cl_neg2.name="138_206-data"

ch_pos1.name="2019-data-Anti"
ch_neg1.name="2019-data"

cl_pos1.name="2019-data-Anti"
cl_neg1.name="2019-data"


cl = pd.concat([cl_pos1,cl_neg1,cl_pos2,cl_neg2],axis=1)
ch = pd.concat([ch_pos1,ch_neg1,ch_pos2,ch_neg2],axis=1)
cl = cl.fillna(0)
newindex=list(cl.index)

import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 1000 #图片像素
plt.rcParams['figure.dpi'] =300 #分
chPicture = ch.plot(title="Frequency statistics",kind="bar",rot=360)
clPicture = cl.plot(kind="bar",title="Peptide length distribution",rot=90)

chPicture = chPicture.get_figure()
clPicture = clPicture.get_figure()
chPicture.savefig("picture\\Peptide_frenquency_4_14")
clPicture.savefig("picture\\Peptide_length_4_14")



