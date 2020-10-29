# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 00:44:47 2019

@author: grqq7
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import glob
import time
from Bio import SeqIO
#from Bio.Alphabet import IUPAC, ProteinAlphabet
from Bio.Seq import Seq
import re

# from repDNA.util import get_data
# seqs =get_data(open(testFile), desc=True)
# for seq in seqs:
    # print seq.seq

###get all fna files in some directory 
#def getFileName(self,path):#读取文件路径
#       path=str(path)
#       filenames=os.listdir(path)
#       subfiles=[path+'\\'+filename for filename in filenames]
#       return subfiles
 
##check protein char  only include 20 chars: ACDEFGHIKLMNPQRSTVWY
def modfastaFiles(tmpdir, suffix):  # 读取文件路径
    ###windows os
    # f = glob.glob(tmpdir + '\\*.' + suffix)
    ###linux os
    f = glob.glob(tmpdir + '/*.' + suffix)
    # fileout = open(dir + '\\' + suffix + '.txt','wt')
    for tmpfile in f:
        # filename = os.path.basename(file)
        # print(tmpfile)
        ##Delte the sequence exclude "The character must be ACDEFGHIKLMNPQRSTVWY"
        # getproteinseq(tmpfile)
        fasta_file = open(tmpfile, "r")
        # print(os.path.splitext(tmpfile)[0])
        modfasta_filename = os.path.splitext(tmpfile)[0] + ".mod" + ".fasta"
        modfasta_file = open(modfasta_filename, "w")
        new_fasta = []
        for seq_record in SeqIO.parse(fasta_file, "fasta"):
            ## ACDEFGHIKLMNPQRSTVWY
            # print(seq_record.seq)
            matchar = re.search(r'[^ACDEFGHIKLMNPQRSTVWY]', str(seq_record.seq), flags=re.I)
            # print(matchar)

            if not matchar:
                # print(seq_record)
                # modfasta_file.write(seq_record)
                new_fasta.append(seq_record)
        if (len(new_fasta) > 0):
            SeqIO.write(new_fasta, modfasta_file, "fasta")
        modfasta_file.close()
        fasta_file.close()

        ##modify the file name
        newname = tmpfile
        os.remove(tmpfile)
        if (os.stat(modfasta_filename).st_size == 0):
            os.remove(modfasta_filename)
        else:
            os.rename(modfasta_filename, newname)


##get all fna files in some directory
def getlibFile(tmpdir,suffix):#读取文件路径
       ###windows os
       #f = glob.glob(tmpdir + '\\*.' + suffix)  
       ###linux os
       f = glob.glob(tmpdir + '/*.' + suffix) 
       #fileout = open(dir + '\\' + suffix + '.txt','wt')  
       for tmpfile in f :  
           #filename = os.path.basename(file)  
           print(tmpfile)        
           featuresFile(tmpfile)
#           

def featuresFile(testFile):    
    print (" Kmer count beging....")
    #testFile='part_long.lib.LINE_vs_SINE.fna'
    #testFile='long.lib.LINE_vs_SINE.fna'
    ##kmer count
    
    #os.system(python nac.pyc testFile Protein Kmer -k 3 -m 1 -out ./testseq/testFile.kmer.out)    
    
    from repDNA.nac import Kmer	
    allkmer = Kmer(k=5, normalize=True, upto=True)
    kmervec = allkmer.make_kmer_vec(open(testFile))
    print (" Kmer count end....")
    #np.savetxt('kmer5.txt',kmervec,fmt='%.4f')
    
    ##revert Kmer
    print ("revert Kmer beging....")
    from repDNA.nac import RevcKmer
    revckmer = RevcKmer(k=5, normalize=True, upto=True)
    revckmervec = revckmer.make_revckmer_vec(open(testFile))
    #np.savetxt('revckmer5.txt',revckmervec,fmt='%.4f') 
    print ("revert Kmer end....")
    
    #####
    ##DAC: Dinucleotide-based auto covariance
    
    print "Dinucleotide-based auto covariance beging...."
    from repDNA.ac import DAC
    dac = DAC(2)
    dacvec = dac.make_dac_vec(open(testFile), all_property=True)
    ##DCC: Dinucleotide-based cross covariance
    from repDNA.ac import DCC
    dcc = DCC(2)
    dccvec = dcc.make_dcc_vec(open(testFile), all_property=True)
    ##DACC: Dinucleotide-based cross covariance
    from repDNA.ac import DACC
    dacc = DACC(2)
    daccvec = dacc.make_dacc_vec(open(testFile), all_property=True)
    ##TAC: Trinucleotide-based auto covariance
    from repDNA.ac import TAC
    tac = TAC(2)
    tacvec = tac.make_tac_vec(open(testFile), all_property=True)
    ##TCC: Trinucleotide-based cross covariance
    from repDNA.ac import TCC
    tcc = TCC(2)
    tccvec = tcc.make_tcc_vec(open(testFile), all_property=True)
    ##TCC: Trinucleotide-based auto-cross covariance
    from repDNA.ac import TACC
    tacc = TACC(2)
    taccvec = tacc.make_tacc_vec(open(testFile), all_property=True)
    
    print ("Dinucleotide-based auto covariance end....")
    
    ###5 Pseudo nucleic acid composition
    print ("5 Pseudo nucleic acid composition beging....")
    ###5.1 Pseudo dinucleotide composition
    from repDNA.psenac import PseDNC
    psednc = PseDNC()
    psedncvec = psednc.make_psednc_vec(open(testFile))
    ####5.2 Pseudo k-tupler composition
    from repDNA.psenac import PseKNC
    pseknc = PseKNC()
    psekncvec = pseknc.make_pseknc_vec(open(testFile))
    ####5.3 Parallel correlation pseudo dinucleotide composition
    from repDNA.psenac import PCPseDNC
    pc_psednc = PCPseDNC()
    pcpsedncvec = pc_psednc.make_pcpsednc_vec(open(testFile),all_property=True)
    ####5.4 Parallel correlation pseudo trinucleotide composition
    from repDNA.psenac import PCPseTNC
    pc_psetnc = PCPseTNC()
    pcpsetncvec = pc_psetnc.make_pcpsetnc_vec(open(testFile),all_property=True)
    ###5.5 Series correlation pseudo dinucleotide composition
    from repDNA.psenac import SCPseDNC
    sc_psednc = SCPseDNC()
    scpsedncvec = sc_psednc.make_scpsednc_vec(open(testFile), all_property=True)
    ####5.6 Series correlation pseudo trinucleotide composition
    from repDNA.psenac import SCPseTNC
    sc_psetnc = SCPseTNC()
    scpsetncvec = sc_psetnc.make_scpsetnc_vec(open(testFile), all_property=True)
    
    print ("5 Pseudo nucleic acid composition end....")
    
    ###all features
    allFeatures = np.hstack((kmervec, revckmervec, dacvec, dccvec, daccvec, tacvec, tccvec, taccvec, psedncvec, psekncvec, pcpsedncvec, pcpsetncvec, scpsedncvec, scpsetncvec))
    FeaFile=testFile+".feats"
    #fileObject = open(FeaFile,'w')
    np.savetxt(FeaFile+'.txt',allFeatures,fmt='%.3f')
    #fileObject.write(allFeatures)
    #fileObject.close()




##main():
st = time.time() 
tmpdir=os.getcwd()
testdir=tmpdir + '\\SCOP167superfamily'
testdir="D:\\light\\BioSeq-Analysis\\data\\ACPP_2015supp"
#print(testdir)

modfastaFiles(testdir,'fasta')

#linesine = "Repbase.lib.LINE_vs_SINE.fna"
#linesine = "Repbase.lib.LTR_vs_nonLTR.fna"
#linesine = "pos-train.f.1.4.fasta"
#
#featuresFile(linesine)
print ("testmodfna run time: "+'%f'%(time.time()-st) )


#print tmpdir
#suffix='fasta'|'fna'
#f = glob.glob(tmpdir + '/*.' + suffix)  
#print f
#       #fileout = open(dir + '\\' + suffix + '.txt','wt')  
#for tmpfile in f :  
#           #filename = os.path.basename(file)  
#    print tmpfile
#  #  featuresFile(tmpfile)










