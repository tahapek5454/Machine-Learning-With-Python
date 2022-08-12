
#1.kutuphaneler
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# veri yukleme
veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# standartlaştırma islemleri yapiliyor veriler bir birine oranl yaklasıyor

x_olcekli = sc.fit_transform(X)
y_olcekli = sc.fit_transform(Y)

# standartlastırma yapmak zorundayız cunku bu metodun aykırı degerlere karsı direnci dusuk

from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf') # guassian kernela denkg geliyor

svr_reg.fit(x_olcekli,y_olcekli)
my_predict = svr_reg.predict(x_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,my_predict,color='blue')
plt.show()











