import pandas as pd


for fea_num in range(1, 20):
    file = "Aver_Result_"+str(feature_num)+".csv"
    filepath = "Result\\Independent_Metric_Second_Layer\\" + file
    data = pd.read_csv(filepath)