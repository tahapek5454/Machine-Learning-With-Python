import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\musteriler.csv')

X = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans ( n_clusters = 3, init = 'k-means++')
#cluster kume sayisi inti ise merkez noktalarının yerlesme metodu
kmeans.fit(X)

print(kmeans.cluster_centers_)
# merkex noktalarının neresinin oldugunu verir
sonuclar = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    # wcss degelerini dondurup dirsek kısmı bularak kac kume olmasını bize yararlı olacagini bulurz

plt.plot(range(1,11),sonuclar)
plt.show()
