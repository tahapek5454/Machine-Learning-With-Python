#import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as   sm

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\maaslar_yeni.csv') # veri okuma

print(veriler)

temp = veriler

print(veriler.corr()) # pd den gelen bir ozellik bize stunlar arasi iliskiyi gosterir

# veriyi inceledigimde id ve unvan kisminin algoritmalari cin gereksiz oldugunu fark ettim
# unvan zaten unvan seviyesi diye numeriklestirilmis
# bu stunlardan kurutlalim

veriler = veriler.drop(['Calisan ID','unvan'], axis=1)

print('*'*50)

print(veriler)

x = veriler.iloc[:,0:1]
y = veriler.iloc[:,[-1]]

X = x.values
Y = y.values

#LinearRegression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

print('Linear Regression OLS')
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())
#burdali p degerine bakararak r-square degerini yukselttik
#ilgili stunlari kaldirdik


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

print('Polynomial Linear Regression degree = 2 OLS')
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

print('Polynomial Linear Regression degree = 4 OLS')
model3 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model3.fit().summary())

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = sc2.fit_transform(Y)


from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

print('SVR OLS')
model4 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model4.fit().summary())



#Decision Tree Regresyon
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4

print('Decision Tree OLS')
model5 = sm.OLS(r_dt.predict(X),X)
print(model5.fit().summary())

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())

print('Random Forest OLS')
model6 = sm.OLS(rf_reg.predict(X),X)
print(model6.fit().summary())



print('*'*50)
print('*'*50)
print('*'*50)

print('Verinin kendisi')
print(temp)
print('*'*50)
print('Secili bir degerin maas tahmini algoritmalara goere')
print('*'*50)

print('Linear Regression 8 seviye icin verdigi Maas')
print(lin_reg.predict([[8]]))
print('*'*50)

print('Polynomial Linear Regression 8 seviye icin verdigi Maas')
print(lin_reg2.predict(poly_reg.fit_transform([[8]])))
print('*'*50)

print('SVR 8 seviye icin verdigi Maas')
print(svr_reg.predict(sc1.fit_transform([[8]])))
print('*'*50)

print('Decision Tree 8 seviye icin verdigi Maas')
print(r_dt.predict([[8]]))
print('*'*50)

print('Random Forest 8 seviye icin verdigi Maas')
print(rf_reg.predict([[8]]))
print('*'*50)



# P değeri bir karşılaştırmada “istatistiksel anlamlı fark vardır” kararı vereceğimiz zaman 
# yapacağımız olası hata miktarını gösterir
# Bir test sonucunda bulunan P değeri 0,05'in altında bir değer ise karşılaştırma sonucunda
# anlamlı farklılık bulunduğu anlamına gelir.








