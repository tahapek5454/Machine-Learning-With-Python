#import lib
import imp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


#1-)Veri yukleme

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\satislar.csv') # veri okuma

aylar = veriler[['Aylar']]


satislar = veriler[['Satislar']]


satislar2 = veriler.iloc[:,:1].values # ekstra farkı anlamak icin yazdım


#verilerin egitim ve test icin bolunmesi


x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)


#verilerin olceklenmesi


sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test  = sc.fit_transform(y_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train,Y_train) # kendini burdaki train verilerine gore ayarlayacak

my_prediction = lr.predict(X_test) # ogrenmis oldugu fit ettigi yapiyi simdi test df e gore tahmin edicek


print('*'*50)
print('My Predict....')
print(my_prediction)
print('*'*50)
print('Real Result....')
print(Y_test)

# Standartasyon olmadan

lr.fit(x_train,y_train) # kendini burdaki train verilerine gore ayarlayacak

my_prediction = lr.predict(x_test) # ogrenmis oldugu fit ettigi yapiyi simdi test df e gore tahmin edicek
print('*'*50)
print('Standartasyon olmadan.......')
print('*'*50)
print('My Predict....')
print(my_prediction)
print('*'*50)
print('Real Result....')
print(y_test)






