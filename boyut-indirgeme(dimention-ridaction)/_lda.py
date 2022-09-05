import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#veri okuma
veriler = pd.read_csv(r"C:\Users\90543\Desktop\VS\machine-leraning-python\datas\wine.csv")
print(veriler)

X = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values


#kume bolumu
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
# kac boyura indirgemek istedigimi belittim
x_train2 = lda.fit_transform(x_train,y_train)
#bu sefer iki parametre vermemiz LDA nin verinin hangi kumede odugunu bilmesi gerektigindendir
#lda sınıfları göztiyor pca gözetmiyor
x_test2 = lda.transform(x_test)
#burda tek veridk cunku zaten ogrendi

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier2 = LogisticRegression(random_state=0)

#lda dan once
classifier.fit(x_train,y_train)

#lda dan sonra
classifier2.fit(x_train2,y_train)

y_predict = classifier.predict(x_test)
y_predict2 = classifier2.predict(x_test2)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
cm2 = confusion_matrix(y_test,y_predict2)


print(cm)
print(cm2)