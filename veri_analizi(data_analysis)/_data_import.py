#import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Veri yukleme

veriler = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\veriler.csv') # veri okuma

print(veriler)

print('*'*50)

boy = veriler['boy']

print(boy)

print('*'*50)

boy_kilo = veriler[['boy','kilo']]

print(boy_kilo)
