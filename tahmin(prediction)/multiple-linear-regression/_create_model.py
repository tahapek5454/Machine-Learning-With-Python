#import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


#1-)Veri yukleme

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\veriler.csv') # veri okuma

print(veriler)

sayisal_veri = veriler.iloc[:,1:4].values # tum satirlar ve 1,2,3 . kolanlardaki tum satirlar gelsin



#3-)Kategorik degerleri numerige cevirme

print('*'*50)

ulke = veriler.iloc[:,0:1].values # ulkeler stunun ayirdik values np yapiyor
print(ulke)

le = preprocessing.LabelEncoder() # encoden nesenesi oluturduk

ulke[:,0] = le.fit_transform(veriler.iloc[:,0:1]) # verilerdeki bilgiler ile fit transform islemi yaparak 
# numerik sayi elde ettik



ohe = preprocessing.OneHotEncoder() # onehotencoder nesnesi olusturduk
ulke = ohe.fit_transform(ulke).toarray() # olustan numerik listeyi onehot olucak sekildi fit transofrm ettik





c = veriler.iloc[:,-1:].values # ulkeler stunun ayirdik values np yapiyor
print(ulke)

print('C dizisi...')
print(c)

le = preprocessing.LabelEncoder() # encoden nesenesi oluturduk

c[:,-1] = le.fit_transform(veriler.iloc[:,-1]) # verilerdeki bilgiler ile fit transform islemi yaparak 
# numerik sayi elde ettik



ohe = preprocessing.OneHotEncoder() # onehotencoder nesnesi olusturduk
c = ohe.fit_transform(c).toarray() # olustan numerik listeyi onehot olucak sekildi fit transofrm ettik

print('one hottan sonra c ')
print(c)

# one hot encoder yapinca dummy variable olustu onu silelim

c = c[:,0:1]

print(c)



#4-)Birlestirme


# dataframe e cevirme
result1 = pd.DataFrame(data=ulke, index=range(22) , columns=['fr','tr','us'])

result2 = pd.DataFrame(data=sayisal_veri, index=range(22) , columns=['boy','kilo','yas'])

result3 = pd.DataFrame(data=c, index=range(22) , columns=['cinsiyet'])

# birlestirme
result4 = pd.concat([result1,result2],axis=1) 
result = pd.concat([result4,result3],axis=1)

print('*'*50)

print(result)

#5-)bolme islemi

x_train , x_test , y_train , y_test = train_test_split(result4,result3,test_size=0.33,random_state=0)

# hedef degiskenin bulundugu ve diger stunlarin bulunudugu frameleri uygun bir sekilde boluyor
# randım_State seed gorevi goruyor

# x in train kısmıyla y nin train kısmı es parcalar davamı yani
# x in test kısmıyla y nin test kısmı es parcalar devamı yani
print('*'*50)

print('x_train...')
print(x_train)
print('*'*50)

print('y_train...')
print(y_train)
print('*'*50)

print('x_test...')
print(x_test)
print('*'*50)

print('y_test...')
print(y_test)
print('*'*50)


# model olusturma

regressor = LinearRegression()

regressor.fit(x_train,y_train)
y_predict = regressor.predict(x_test)

print('*'*50)
print('*'*50)
print('y_test...')
print(y_test)
print('*'*50)
print('y_predict...')
print(pd.DataFrame(y_predict,columns=['cinsiyet']))


# 2 . deneme boyu tahmin etmeye calisalim

print('*'*50)
print('Boy tahmini')
print('*'*50)

boy = result.iloc[:,3:4]
print('Boy')
print(boy)

print('*'*50)

sol = result.iloc[:,0:3]

print('*'*50)
print('Sol')
print(sol)

sag = result.iloc[:,4:]   # boyun sol ve sag kisimlarini aldik

print('*'*50)
print('sag')
print(sag)

features = pd.concat([sol,sag],axis=1)

x_train , x_test , y_train , y_test = train_test_split(features,boy,test_size=0.33,random_state=0)

regressor.fit(x_train,y_train)
y_predict = regressor.predict(x_test)

print('Y tahmin...')
print(pd.DataFrame(y_predict,columns=['Boy']))
print('*'*50)
print('Y test')
print(y_test)










