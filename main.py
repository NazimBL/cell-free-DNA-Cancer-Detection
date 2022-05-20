import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import loadtxt

"""""

df = pd.read_csv("cfdna_dataset.csv")
del df["Patient"]
del df["Timepoint"]
df = df.replace(["Y","N"],[1,0])
#df['Patient Type'] = 1

df = df.replace(["Healthy"],[0])
df = df.replace(["Breast Cancer", "Cholangiocarcinoma", "Colorectal Cancer"
                , "Gastric cancer", "Lung Cancer", "Ovarian Cancer"
                ,  "Pancreatic Cancer", "Bile Duct Cancer", "Gastric Cancer"],[1,1,1,1,1,1,1,1,1])

df['Percent Mapped to Target Regions'] = df['Percent Mapped to Target Regions'].str.rstrip('%').astype('float') / 100.0
df.to_csv("data.csv", index=False)
print(df)
"""
dataset = loadtxt('data.csv', delimiter=",")
# split data into X and y
X = dataset[:,1:10]
Y = dataset[:,0]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

#print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))