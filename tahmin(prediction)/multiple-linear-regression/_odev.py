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

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\odev_tenis.csv') # veri okuma

print(veriler)




temprature = veriler.iloc[:,[1]].values # array olarak
label_humidity = veriler.iloc[:,[2]].values # array olarak



#3-)Kategorik degerleri numerige cevirme

veriler2 = veriler.drop(['outlook','temperature','humidity'],axis=1) # busekilde sadece elimizde kategorik kaldı

print(veriler2)

print('*'*50)

outlook = veriler.iloc[:,[0]].values # outlook stunun ayirdik values np yapiyor
# not one hot encoding islemi uygulanacak

le = preprocessing.LabelEncoder() # encoden nesenesi oluturduk

outlook[:,0] = le.fit_transform(veriler.iloc[:,0:1]) # verilerdeki bilgiler ile fit transform islemi yaparak 
# numerik sayi elde ettik

print('OUTLOOK')
print(outlook)

veriler2 = veriler2.apply(le.fit_transform)

print('Encoded datas')
print(veriler2)



print('*'*50)





ohe = preprocessing.OneHotEncoder() # onehotencoder nesnesi olusturduk
outlook = ohe.fit_transform(outlook).toarray() # olustan numerik listeyi onehot olucak sekildi fit transofrm ettik




print('onehot outlook')
print(outlook)







#4-)Birlestirme


# dataframe e cevirme
result1 = pd.DataFrame(data=outlook , columns=['overcast','rainy','sunny'])

print(result1)

result2 = pd.DataFrame(data=temprature, columns=['temprature'])

print(result2)

result3 = pd.DataFrame(data=veriler2 , columns=['windy','play'])

print(result3)


result5_label = pd.DataFrame(data=label_humidity , columns=['humidity'])

print(result5_label)

# birlestirme
feature = pd.concat([result1,result2],axis=1) 
feature = pd.concat([feature,result3],axis=1)

totalFrame = pd.concat([feature,result5_label],axis=1)

print('Birlestirme isleminden sonra ...')
print('Feature')
print(feature)
print('*'*50)
print('Label')
print(result5_label)


print('*'*50)



#5-)bolme islemi

x_train , x_test , y_train , y_test = train_test_split(feature,result5_label,test_size=0.33,random_state=0)

# hedef degiskenin bulundugu ve diger stunlarin bulunudugu frameleri uygun bir sekilde boluyor
# randım_State seed gorevi goruyor

# x in train kısmıyla y nin train kısmı es parcalar davamı yani
# x in test kısmıyla y nin test kısmı es parcalar devamı yani
print('*'*50)



# model olusturma

regressor = LinearRegression()

regressor.fit(x_train,y_train)
y_predict = regressor.predict(x_test)


print('Y_ test')
print(y_test)

print('y_predict')
print(pd.DataFrame(y_predict))





# backward_elimination

x = np.append(arr = np.ones((14,1)).astype(int) , values=feature , axis=1) # bu sabit degiskeni ekliyoruz

# values da diger verileri ekliyoruz

x_l = feature.iloc[:,[0,1,2,3,4,5]].values # p degeri 0.05 den buyukleri cikar
x_l = np.array(x_l,dtype=float)
new_model = sm.OLS(result5_label,x_l).fit()
print(new_model.summary())


# bir tane ele islemi yapalım ornek icin

x_l = feature.iloc[:,[0,1,2,3,5]].values # p degeri 0.05 den buyukleri cikar
x_l = np.array(x_l,dtype=float)
new_model = sm.OLS(result5_label,x_l).fit()
print(new_model.summary())


regressor.fit(x_train.iloc[:,[0,1,2,3,5]],y_train)
y_predict = regressor.predict(x_test.iloc[:,[0,1,2,3,5]])


print('Y_ test')
print(y_test)

print('y_predict')
print(pd.DataFrame(y_predict)) # iyilestirdik






