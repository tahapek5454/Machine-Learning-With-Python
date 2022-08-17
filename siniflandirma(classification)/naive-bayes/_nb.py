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



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(xs_train,y_train)
y_predict = gnb.predict(xs_test)
#gaussiannb bize bize reel sonucu olusabilecek ciktilar icin kullanılır
#Multinomial Bayes bize int sayi gibi dusunebilir mesela universiteler 1,2,3,4
#bernouli navie bayes ise bernolliden cikartirsak iki sonuc dondurucek evet hayır kadın erkek gibi



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
print('GaussianNB')
print(cm)
# kosegen bize dogru tahmini verir bizde yok












