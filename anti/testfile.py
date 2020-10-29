import pandas as pd

data = pd.read_csv("Catboost_Rank\\Catboost_Ave_Rank.csv",index_col=0,header=0)
data = data.sort_values(by="Acc",ascending=False)
print(data)