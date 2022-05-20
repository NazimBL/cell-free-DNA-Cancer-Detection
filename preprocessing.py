import numpy as np
import pandas as pd

df = pd.read_csv("cfdna_dataset.csv")
del df["Patient"]
del df["Timepoint"]
df = df.replace(["Y","N"],[1,0])
#df['Patient Type'] = 1

df = df.replace(["Healthy"],[0])
df = df.replace(["Breast Cancer", "Cholangiocarcinoma", "Colorectal Cancer"
                , "Gastric cancer", "Lung Cancer", "Ovarian Cancer"
                ,  "Pancreatic Cancer"],[1,1,1,1,1,1,1])

df['Percent Mapped to Target Regions'] = df['Percent Mapped to Target Regions'].str.rstrip('%').astype('float') / 100.0
df.to_csv("data.csv", index=False)
