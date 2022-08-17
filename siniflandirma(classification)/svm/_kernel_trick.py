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



from sklearn.svm import SVC
svc = SVC(kernel='poly')
# farklı kernel yapilarina dokumantasyonlardan ulasabilirz
# farklı yapıları da deneyerek verimize en uygun makine ogrenmesi yontemini bulmaliyiz
# birden fazla kernel eklenebilir veya yontemş degistirebilir verinin duzenine gore 
svc.fit(xs_train,y_train)
y_predict = svc.predict(xs_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
print('SVC')
print(cm)
# kosegen bize dogru tahmini verir bizde yok












