#import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Not sınıflandırma yontemi genelde kategorik veri elde etmek istedigimizde kulanılır
# Burda kadın mı erkek mi onu bulmaya calisiyoruz

#1-)Veri yukleme

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\veriler.csv') # veri okuma

print(veriler)

x = veriler.iloc[:,1:-1].values
y = veriler.iloc[:,[-1]].values


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.33,random_state=0)

#6-) Standartasyon
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# standartlaştırma islemleri yapiliyor veriler bir birine oranl yaklasıyor
xs_train = sc.fit_transform(x_train)
xs_test = sc.transform(x_test)
# Not fit _ transform ogren ve ogrendigini uygula demek x_train icin ogrendi zate
# Bu yüzüden x_test icin sadece transfırm ediyoruz 

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1 , metric='minkowski')
# knn verileri komsularına bakarak sınıflandırmaya yarayan bir yontem 
# n sayısına 5 vermemiz bize bir veriyi tahmin ederken en yakın 5 veriye bakacagini
# ve ona gore tahmin edecegini belirtir
# metric ise verilerin arasındaki mesafeyi hangi yontemle bulacagina karar ver,r
# komsu sayisini 5 yaptigimzda burda 1 tane dogru tahmin yaptı
# bir de komsu sayisi 1 yapalım
# komsu sayisini elimizdeki probleme gore belirlememiz lazım
# komsu sayisi 1 iken 7 dogru yaptı

knn.fit(xs_train,y_train)

y_predict = knn.predict(xs_test)

print('Y_test')
print(pd.DataFrame(y_test))
print('Y_predict')
print(pd.DataFrame(y_predict))

# Tam tersi tahmin etti :)

# Bu yontemin dogurulunu test etmek icin su sekilde bir yontem izleriz

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
print(cm)
# kosegen bize dogru tahmini verir bizde yok












