import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


X=np.zeros((12,100))
y=np.zeros(12)

for i in range(1,12):
    f='photo/'+str(i)+'.jpg'
    #print('\n',f)
    #print(type(f))
    
    I=cv2.imread(f)
    I=cv2.resize(I,(100,100))
    I=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    #print('shape=',np.shape(I))
    I=np.sum(I,axis=0)
    I=np.matrix(I)
    
    
    X[i-1,:]=I
    y[i-1]=i
    
print('shape of x=',np.shape(X))
print('shape of y=',np.shape(y))


print("\n-----------LR--------------")

from sklearn.linear_model import LinearRegression
mdl=LinearRegression()
mdl.fit(X,y)
print("\nmdl.score(x,y)=",mdl.score(X,y)*100,"%")


print("\n-----------KNN--------------")
from sklearn.neighbors import KNeighborsClassifier
mdl=KNeighborsClassifier(n_neighbors=3)


mdl.fit(X,y)
print("\nmdl1.score(x,y)=",mdl.score(X,y)*100,"%")

print("\n-----------NB--------------")
from sklearn.naive_bayes import GaussianNB
mdl=GaussianNB()


mdl.fit(X,y)

print("\nmdl.score(x,y)=",mdl.score(X,y)*100,"%")


I=cv2.imread('photo/4.jpg')
I=cv2.resize(I,(100,100))
I=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
#print('shape=',np.shape(I))
I=np.sum(I,axis=0)
I=np.matrix(I)

result=mdl.predict(I)
result=np.floor(result)
print("result = ",result)


if 1<=result<=4:
    print('It is a fan')
elif 5<=result<=8:
    print('It is a table')
elif 9<=result<=12:
    print("It is a chair")


