import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier 
titanic=sns.load_dataset('titanic')
titanic.shape
titanic=titanic[['survived','pclass','sex','age']]
titanic=titanic.dropna(axis=0)
titanic['sex'].replace(['male','female'] , [0,1],inplace=True)
h=titanic.head()
print(h)
model= KNeighborsClassifier()
y=titanic['survived']
X=titanic.drop('survived',axis=1)
print(y)
print(X)
f=model.fit(X,y) 
s=model.score(X,y)
print(s)
def survie(model,pclass=3,sex=0,age=21):
    X=np.array([pclass,sex,age]).reshape(1,3)
    print(model.predict_proba(X)) #proba appartenance classe 0 ou 1
    print(model.predict(X)) #survived or no
survie(model) 



