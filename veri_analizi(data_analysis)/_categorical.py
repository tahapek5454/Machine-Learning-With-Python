#import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing 


#Veri yukleme

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\eksikveriler.csv') # veri okuma

print(veriler)

#Nan verilere deger atama

imputer = SimpleImputer(missing_values=np.nan , strategy='mean') # neyle degeri değiştireceğini belirler

sayisal_veri = veriler.iloc[:,1:4].values # tum satirlar ve 1,2,3 . kolanlardaki tum satirlar gelsin


print(sayisal_veri)

imputer = imputer.fit(sayisal_veri[:,0:3]) # tum satileri ve 0 dan 3 . indexe kadaraki stunlarla eğit
sayisal_veri[:,0:3] = imputer.transform(sayisal_veri[:,0:3]) # eğitimi isle boslari doldur ortalamaya gore

# not parametre olarak [:,0:3] vermeseydik de olurdu cunku kompile degisiklik yapiyoruz
# ilgili bos veriyi kendi stunundaki diger degerlerin ortalamasini koyuyor ona ayarladık

print('*'*50)

print(sayisal_veri)

#Kategorik degerleri numerige cevirme

print('*'*50)

ulke = veriler.iloc[:,0:1].values # ulkeler stunun ayirdik
print(ulke)

le = preprocessing.LabelEncoder() # encoden nesenesi oluturduk

ulke[:,0] = le.fit_transform(veriler.iloc[:,0:1]) # veriler dek bilgiler ile fit transform islemi yaraka 
# numerik sayi elde ettik

print(ulke)

ohe = preprocessing.OneHotEncoder() # onehotencoder nesnesi olusturduk
ulke = ohe.fit_transform(ulke).toarray() # olustan numerik listeyi onehot olucak sekildi fit transofrm ettik
print(ulke)



