from clean_data import *


"""This is the linear support vector classification algorithm. We declare the algorithm, train+split, fit, predict."""
from sklearn.svm import LinearSVC
lin_svc=LinearSVC()

from sklearn.model_selection import train_test_split
Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)

lin_svc.fit(Data_train, Labels_train)
print (f'Lin_svc accuracy: {lin_svc.score(Data_test, Labels_test)}')



"""This is the k-nearest-neighbors classification algorithm. We declare the algorithm (with k=19), train+split, fit, predict."""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=19)

from sklearn.model_selection import train_test_split
Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)

knn.fit(Data_train, Labels_train)
print (f'knn accuracy: {knn.score(Data_test, Labels_test)}')



"""Visualization of results"""
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

array=normalize(confusion_matrix(Labels_test, knn.predict(Data_test)))
  
df_cm = pd.DataFrame(array, range(17),
                  range(17))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 10}, cmap='Greens')# font size
plt.xlabel('Predicted label', fontsize=16)
plt.ylabel('True label', fontsize=16)
plt.show()