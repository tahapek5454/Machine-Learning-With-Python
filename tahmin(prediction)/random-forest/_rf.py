#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# veri yukleme
veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


from sklearn.ensemble import RandomForestRegressor

rg_reg = RandomForestRegressor(n_estimators=10,random_state=0)
# n_esimatorss bize kac tane decision tree'den baglanacagini belirtiyor
rg_reg.fit(X,Y.ravel())
my_predict = rg_reg.predict(X)

print(rg_reg.predict([[6.6]]))

plt.scatter(X,Y,color='red')
plt.plot(X,my_predict,color='blue')
plt.show()

# Birden fazla decision tree ile islem yapiyor