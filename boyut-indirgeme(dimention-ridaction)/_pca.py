import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#veri okuma
veriler = pd.read_csv(r"C:\Users\90543\Desktop\VS\machine-leraning-python\datas\wine.csv")
print(veriler)

X = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values


#kume bolumu
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
# kac stuna indirgememiz gerektigini giriyoruz
x_train2 = pca.fit_transform(x_train)
x_test2 = pca.transform(x_test)

from sklearn.linear_model import LogisticRegression
#pca donusumunden once
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

#pca donusumunden sonra
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(x_train2,y_train)

#tahminler

y_predic = classifier.predict(x_test)
y_predic2 = classifier2.predict(x_test2)

#karsilastimra
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predic)
cm2 = confusion_matrix(y_test,y_predic2)


print(cm)
print('*'*50)
print(cm2)