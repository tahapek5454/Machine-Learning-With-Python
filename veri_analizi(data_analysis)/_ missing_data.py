#import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer 


#Veri yukleme

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\eksikveriler.csv') # veri okuma

print(veriler)

imputer = SimpleImputer(missing_values=np.nan , strategy='mean') # neyle degeri değiştireceğini belirler

sayisal_veri = veriler.iloc[:,1:4].values # tum satirlar ve 1,2,3 . kolanlardaki tum satirlar gelsin


print(sayisal_veri)

imputer = imputer.fit(sayisal_veri[:,0:3]) # tum satileri ve 0 dan 3 . indexe kadaraki stunlarla eğit
sayisal_veri[:,0:3] = imputer.transform(sayisal_veri[:,0:3]) # eğitimi isle boslari doldur ortalamaya gore

# not parametre olarak [:,0:3] vermeseydik de olurdu cunku kompile degisiklik yapiyoruz
# ilgili bos veriyi kendi stunundaki diger degerlerin ortalamasini koyuyor ona ayarladık

print('*'*50)

print(sayisal_veri)


