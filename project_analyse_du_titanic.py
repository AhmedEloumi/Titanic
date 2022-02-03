import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_excel('C:/Users/lenovo/Desktop/Auto F/Formation python machine learning/titanic3 (3).xls')
print(data.shape)
print(data.columns)
print(data.head()) #eqv à data.shape
#eliminer columns
data=data.drop(['name','sibsp', 'parch', 'ticket','fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest'],axis=1) 
print(data.columns)
print(data.head())
print(data.describe())
#on peut ajouter un ages par defaut(moyen d'age) des datas qui on a 
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





