#Import some basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection,neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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
clf = neighbors.KNeighborsClassifier()
knn_clf = clf.fit(x_train,y_train)

y_predict = knn_clf.predict(x_test)
print('predicted chances of flood')
print(y_predict)

print("actual values of floods:")
print(y_test)

knn_accuracy = cross_val_score(knn_clf,x_test,y_test,cv=3,scoring='accuracy',n_jobs=-1)
knn_accuracy.mean()

#Logistic Regression
x_train_std = minmax.fit_transform(x_train)
x_test_std = minmax.transform(x_test)

lr = LogisticRegression()
lr_clf = lr.fit(x_train_std,y_train)
y_predict = lr_clf.predict(x_test_std)
print('Predicted chances of flood')
print(y_predict)
print('Actual chances of flood')
print(y_test.values)

print("\naccuracy score: %f"%(accuracy_score(y_test,y_predict)*100))
print("recall score: %f"%(recall_score(y_test,y_predict)*100))
print("roc score: %f"%(roc_auc_score(y_test,y_predict)*100))
lr_accuracy = cross_val_score(lr_clf,x_test_std,y_test,cv=3,scoring='accuracy',n_jobs=-1)

#Decision Tree classification
dtc_clf = DecisionTreeClassifier()
dtc_clf.fit(x_train,y_train)
dtc_clf_acc = cross_val_score(dtc_clf,x_train_std,y_train,cv=3,scoring="accuracy",n_jobs=-1)
dtc_clf_acc

y_pred = dtc_clf.predict(x_test)
print(y_pred)

print("actual values :")
print(y_test.values)
print("\naccuracy score:%f"%(accuracy_score(y_test,y_pred)*100))
print("recall score:%f"%(recall_score(y_test,y_pred)*100))
print("roc score:%f"%(roc_auc_score(y_test,y_pred)*100))

#random forest
from sklearn.ensemble import RandomForestClassifier
rmf = RandomForestClassifier(max_depth=3,random_state=0)
rmf_clf = rmf.fit(x_train,y_train)
rmf_clf
rmf_clf_acc = cross_val_score(rmf_clf,x_train_std,y_train,cv=3,scoring="accuracy",n_jobs=-1)
#rmf_proba = cross_val_predict(rmf_clf,x_train_std,y_train,cv=3,method='predict_proba')
rmf_clf_acc
y_pred = rmf_clf.predict(x_test)
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
print("\naccuracy score:%f"%(accuracy_score(y_test,y_pred)*100))
print("recall score:%f"%(recall_score(y_test,y_pred)*100))
print("roc score:%f"%(roc_auc_score(y_test,y_pred)*100))

