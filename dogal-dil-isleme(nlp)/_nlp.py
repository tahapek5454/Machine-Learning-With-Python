import numpy as np
import pandas as pd

yorumlar = pd.read_csv(r'C:\Users\90543\Desktop\VS\machine-leraning-python\datas\yorumlar.csv')

print(yorumlar)

import re

#noktalama isaretlerini kaldiracagiz
yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][0])
print(yorum)