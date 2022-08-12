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
my_predict = dt_reg.predict(X)

plt.scatter(X,Y,color='red')
plt.plot(X,my_predict,color='blue')
plt.show()

# Decision Tree aslinda verileri belli ozelleklirine gore kumelere ayiriyor ve o kumelere icindeki
# verilerin ortalamasini veriyor bizim tahmin etmek istedigimiz veriyi hanki kumenin icersindeyse
# bize o kumedeki degeri vermis oluyor 