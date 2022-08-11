#import lib
from re import X
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


#1-)Veri yukleme

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\maaslar.csv') # veri okuma

print(veriler)

# veride null degerler yok ekleme islemi yapmamiza gerek yok
# veride kullanacagimiz kategorik degisken yok encoder yapmyacagiz
# veriyi parcalamayacagimizdan birlesim islemine gerek yok
# tum veriyi traine sokacagimizdan test ve train diye bolmeye ihtiyacımız yok

# veriyi x ve y olarak bolelim label feature gibi


# Not dataframe sklearn icinde kullanırken problem cikartabiliyor
# o yuzden np arrayi yani valuesleri atıcaz
# ben dataframe ile de denedim oldu aslinda ama kucuk bir bilgi kalsın

# kordinatlara ayirma islemi

x = veriler.iloc[:,[1]]
y = veriler.iloc[:,-1:]

x_array = x.values
y_array = y.values

# polinomlar farkını gorme acisindan linear regression denemesi

lin_reg = LinearRegression()

lin_reg.fit(x_array,y_array)

plt.scatter(x_array,y_array,color = 'red')
plt.plot(x_array,lin_reg.predict(x_array),color='green')
# plt.show()

# polynomial regressin baslangici

poly_reg = PolynomialFeatures(degree=2) # polinomumuz 2 . dereceden olsun dedik
x_poly = poly_reg.fit_transform(x_array) # x arrayini kendi sistemine benzet fit_tranform et
print(x_poly)
# bu kalıbımı linear regression uzerine uygulayacagiz
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_array) # yapmis oldugumuz x_poly kalıbıyla y_arrayi bagdastir fit et

plt.scatter(x_array,y_array,color='red')
plt.plot(x_array,lin_reg2.predict(x_poly),color='blue')
# plt.show()

# bir de dereceyi arttirarak model olusturalim

poly_reg = PolynomialFeatures(degree=4) # polinomumuz 2 . dereceden olsun dedik
x_poly = poly_reg.fit_transform(x_array) # x arrayini kendi sistemine benzet fit_tranform et
print(x_poly)
# bu kalıbımı linear regression uzerine uygulayacagiz
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_array) # yapmis oldugumuz x_poly kalıbıyla y_arrayi bagdastir fit et

plt.scatter(x_array,y_array,color='red')
plt.plot(x_array,lin_reg2.predict(x_poly),color='yellow')
plt.show()











