import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_excel('C:/Users/lenovo/Desktop/Auto F/Formation python machine learning/titanic3 (3).xls')
print(data.shape)
print(data.columns)
print(data.head())
#indexer le tableau avec le nom des passagers
#data=data.set_index('name')
#print(data['age'])
#eqv à data.shape
#eliminer columns
data=data.drop(['name','sibsp', 'parch', 'ticket','fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest'],axis=1) 
print(data.columns)
print(data.head())
print(data.describe())
#on peut ajouter un ages par defaut(moyen d'age) des passagers qui on a 
#pas une information sur leurs ages
#data.fillna(data['age'].mean())
#peut modifier la réalité
#eliminer_lignes(des datas qui on a pas une information sur leurs ages )
data=data.dropna(axis=0)
print(data.shape)
print(data.describe())
#classes
print(data["pclass"].value_counts())
#graphique
#print(data["pclass"].value_counts().plot.bar())
#repartition des ages avec histogramme
print(data['age'].hist())
#groupant les gens seloon leur sexe
print(data.groupby(['sex']).mean())
print(data.groupby(['sex']).count())
print(data.groupby(['sex']).max())
print(data.groupby(['sex','pclass']).mean())
#indexing
print(data['age'][0:10])
passagers_mineurs=data[data['age']<18]
print(passagers_mineurs['pclass'].value_counts())
print(passagers_mineurs.groupby(['sex','pclass']).mean())
#index_localisation
print(data.iloc[0:2,0:2])
print(data.loc[0:2,['age','sex']])
#donner le nombre de passagers selon leurs ages 
#1ére méthode

data.loc[data['age'] <=20 ,'age']=0
data.loc[(data['age']<=30) & (data['age']>20) ,'age']=1
data.loc[(data['age']<=40 ) & (data['age']>30) ,'age']=2
data.loc[data['age']>40 ,'age']=3
print(data['age'].value_counts())
print(data.groupby(['age']).mean())

#2éme methode (map)!! 
print(data["age"].map(lambda x:x+1))
def category_ages(age):
    if age<=20:
        return '-20 ans'
    elif (age>20)&(age<=30):
        return '20-30 ans'
    elif (age>30)&(age<=40):
        return '30-40 ans'
    else: 
        return '+40 ans' 
print(data["age"].map(category_ages))
#convertir les données
a=data['sex'].map({'male':0,'female':1})
print(a)
#ou bien
b=data['sex'].replace(['male','female'],[0,1])
print(b)
#ou bien
c=data['sex'].astype('category').cat.codes
print(c)