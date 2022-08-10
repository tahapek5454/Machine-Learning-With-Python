#import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as   sm


#1-)Veri yukleme

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\veriler.csv') # veri okuma

print(veriler)

sayisal_veri = veriler.iloc[:,1:4].values # tum satirlar ve 1,2,3 . kolanlardaki tum satirlar gelsin



#3-)Kategorik degerleri numerige cevirme

print('*'*50)

ulke = veriler.iloc[:,0:1].values # ulkeler stunun ayirdik values np yapiyor


le = preprocessing.LabelEncoder() # encoden nesenesi oluturduk

ulke[:,0] = le.fit_transform(veriler.iloc[:,0:1]) # verilerdeki bilgiler ile fit transform islemi yaparak 
# numerik sayi elde ettik



ohe = preprocessing.OneHotEncoder() # onehotencoder nesnesi olusturduk
ulke = ohe.fit_transform(ulke).toarray() # olustan numerik listeyi onehot olucak sekildi fit transofrm ettik





c = veriler.iloc[:,-1:].values # ulkeler stunun ayirdik values np yapiyor



le = preprocessing.LabelEncoder() # encoden nesenesi oluturduk

c[:,-1] = le.fit_transform(veriler.iloc[:,-1]) # verilerdeki bilgiler ile fit transform islemi yaparak 
# numerik sayi elde ettik



ohe = preprocessing.OneHotEncoder() # onehotencoder nesnesi olusturduk
c = ohe.fit_transform(c).toarray() # olustan numerik listeyi onehot olucak sekildi fit transofrm ettik



# one hot encoder yapinca dummy variable olustu onu silelim

c = c[:,0:1]





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


# model olusturma

regressor = LinearRegression()

regressor.fit(x_train,y_train)
y_predict = regressor.predict(x_test)



# 2 . deneme boyu tahmin etmeye calisalim



boy = result.iloc[:,3:4]


print('*'*50)

sol = result.iloc[:,0:3]


sag = result.iloc[:,4:]   # boyun sol ve sag kisimlarini aldik



features = pd.concat([sol,sag],axis=1)

x_train , x_test , y_train , y_test = train_test_split(features,boy,test_size=0.33,random_state=0)

regressor.fit(x_train,y_train)
y_predict = regressor.predict(x_test)

# backward_elimination

print('onceki features degeri')
print(features)

x = np.append(arr = np.ones((22,1)).astype(int) , values=features , axis=1) # bu sabit degiskeni ekliyoruz

# values da diger verileri ekliyoruz

x_l = features.iloc[:,[0,1,2,3,4,5]].values # p degeri 0.05 den buyukleri cikar
x_l = np.array(x_l,dtype=float)
new_model = sm.OLS(boy,x_l).fit()
print(new_model.summary())


x_l = features.iloc[:,[0,1,2,3,5]].values
x_l = np.array(x_l,dtype=float)
new_model = sm.OLS(boy,x_l).fit()
print(new_model.summary())

#bu da demek oluyor ki yeni featuremes ustteki olmali ayarlayalım ve tekrar deneyelim

new_features = features.iloc[:,[0,1,2,3,5]].values

x_train , x_test , y_train , y_test = train_test_split(new_features,boy,test_size=0.33,random_state=0)

regressor.fit(x_train,y_train)
y_predict = regressor.predict(x_test)

print('Y_test')
print(y_test)
print('New Y predict')
print(y_predict)









