import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\musteriler.csv')

X = veriler.iloc[:,3:].values

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
#n_cluster kume sayisi
#affinity iki veri arası mesafe yontemi
#linkage kumeler arasi mesafe yontemi
y_tahmin = ac.fit_predict(X)
#burada hem modeli ogrendik ve predict ile ogrendigimiz modeldeki verilerin kumelerini bastirdik
plt.title('HC')
plt.scatter(X[y_tahmin==0,0],X[y_tahmin==0,1],s=100,c='red')
plt.scatter(X[y_tahmin==1,0],X[y_tahmin==1,1],s=100,c='blue')
plt.scatter(X[y_tahmin==2,0],X[y_tahmin==2,1],s=100,c='green')
# burda numpy array in ozelliklerini kullandik 
# 0,1,2 inci kumeler icin tabloya arrayin 0 ve 1 inci  indexlerini bastir ve renkleri ver
plt.show()


#dendrogram icin
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()

