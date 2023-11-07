#Import some basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection,neighbors
from sklearn.model_selection import train_test_split
data = pd.read_csv(r"C:\Users\khyat\Downloads\kerala.csv")
data.head()

data.apply(lambda x:sum(x.isnull()), axis=0)
data['FLOODS'].replace(['YES','NO'],[1,0],inplace=True)
data.head()

x = data.iloc[:,1:14]
x.head()

y = data.iloc[:, -1]
y.head()

c = data[['JUN','JUL','AUG','SEP']]
c.hist()

plt.show()
minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
minmax.fit(x).transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

x_train.head()
y_train.head()

#KNN Algorithms