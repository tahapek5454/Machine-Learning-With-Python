#1.kutuphaneler
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# veri yukleme
veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,Y)
# Simdi ogrenen modelimizi diske kaydedelim ve tekrar tekrar
# Modeli fit etmeden trahmin isemlerimizi yapalım
import pickle
dosya_yolu = "model_kayit"
pickle.dump(dt_reg,open(dosya_yolu,"wb")) #binary modunda dosyaya modeli kaydettik
yuklenen_model = pickle.load(open(dosya_yolu,"rb")) #binarymodunda modeli okudum

# simdi predict islemini okunan modelden yaacagim
my_predict = yuklenen_model.predict(X)

plt.scatter(X,Y,color='red')
plt.plot(X,my_predict,color='blue')
plt.show()

# Decision Tree aslinda verileri belli ozelleklirine gore kumelere ayiriyor ve o kumelere icindeki
# verilerin ortalamasini veriyor bizim tahmin etmek istedigimiz veriyi hanki kumenin icersindeyse
# bize o kumedeki degeri vermis oluyor 
