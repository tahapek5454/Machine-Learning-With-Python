import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veri = pd.read_csv(r"C:\Users\90543\Desktop\VS\machine-leraning-python\datas\xg.csv")

X = veri.iloc[:,3:13].values
y = veri.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
ohe = ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],remainder='passthrough')
le1 = LabelEncoder()
X[:,1] = le1.fit_transform(X[:,1])
le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])
X = ohe.fit_transform(X)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train,y_train)

y_prediction = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_prediction)

print(cm)