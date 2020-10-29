import os


def identity_compute(Psiblastpath, Filepath, Db_path):
    os.system(" psiblast -query PaCRISPR_Independent_Positive_26.fasta -db ACR_DB -out test.txt -num_iterations 3 -outfmt 6 ")

Psiblastpath = ""
Filepath =""
Db_path =""