#import lib
from re import X
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#1-)Veri yukleme

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\satislar.csv') # veri okuma

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

satislar2 = veriler.iloc[:,:1].values # ekstra farkı anlamak icin yazdım
print(satislar2)

#verilerin egitim ve test icin bolunmesi


x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

print(x_train)
print('*'*50)
print(y_train)

#verilerin olceklenmesi


sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
print('*'*50)
print(X_train)
print('*'*50)
print(X_test)




