# -*- coding: utf-8 -*-
from Bio.Seq import Seq
import numpy as np
import os
import glob
import time
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
#from Bio.Alphabet import IUPAC, ProteinAlphabet
from Bio.Seq import Seq
import re
import pandas as pd
from Bio.Seq import Seq
tmpdir = "E:\\data-AVP"
suffix = "xlsx"
f = glob.glob(tmpdir + '/*.' + suffix)
for filepath in f:
	data = pd.read_excel(filepath,sheet_name=None,index_col=0,header=0)
	data = data['Sheet1']
	path,name=os.path.split(filepath)
	modfasta_filename = "E:\\data-AVP\\FASTA\\"+name[0:len(name)-5]+".fasta"
	print (name)
	new_fasta=[]
	modfasta_file = open(modfasta_filename, "w",encoding="utf-8")
	for index,row in data.iterrows():
		simple_seq=Seq(row['Sequence'])
		simple_seq_r=SeqRecord(simple_seq)
		simple_seq_r.id=str(index)
		simple_seq_r.description = "hello"
		new_fasta.append(simple_seq_r)
	SeqIO.write(new_fasta, modfasta_file, "fasta")

