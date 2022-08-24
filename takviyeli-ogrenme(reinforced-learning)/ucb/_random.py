import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\Ads_CTR_Optimisation.csv')
print(veriler)

import random as rd

n = 10000
d= 10
toplam = 0
secilenler = []

for i in range(0,n):
    ad = rd.randrange(d)
    secilenler.append(ad)
    odul =  veriler.values[i,ad] # satir ve stundaki dger neyse o puan olarak eklenecek
    toplam = toplam+odul


print(toplam)
plt.hist(secilenler)
plt.show()