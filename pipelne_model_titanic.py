import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder,Binarizer
from sklearn.pipeline import make_pipeline,make_union
from sklearn.compose import make_column_transformer ,make_column_selector
from sklearn.impute import SimpleImputer

import seaborn as sns
titanic=sns.load_dataset('titanic')
titanic=titanic[['survived','pclass','sex','age','deck','fare','alone']]
titanic=titanic.dropna(axis=0)
h=titanic.head()
print(h)
y=titanic['survived']
X=titanic.drop('survived',axis=1)
#make_column_transformer indiquer les variables elli theb transformihom
numerical_features=['pclass','age','fare']
categorical_fetures=['sex','deck','alone']
#traiter les valeurs numériques 
numerical_pipeline=make_pipeline(SimpleImputer(),
                                 StandardScaler())
#traiter les valeurs catégoriques
#SimpleImputer enlever les valeurs manquantes ou remplacer les valeurs manquantes par les valeure les plus fréquentes
categorical_pipeline=make_pipeline(SimpleImputer(strategy='most_frequent'),
                                   OneHotEncoder())
#assembler les pipelines
preprocessor=make_column_transformer ((numerical_pipeline,numerical_features),
                                     (categorical_pipeline,categorical_fetures))
model= make_pipeline(preprocessor,SGDClassifier())
f=model.fit(X,y)
print(f)
######################
print("another")
#make_column_selector indiquer les types de variables 

titanic=sns.load_dataset('titanic')

numerical_features=make_column_selector(dtype_include=np.number)
categorical_features=make_column_selector(dtype_exclude=np.number)

numerical_pipeline=make_pipeline(SimpleImputer(),
                                 StandardScaler())

categorical_pipeline=make_pipeline(SimpleImputer(strategy='most_frequent'),
                                   OneHotEncoder())

preprocessor=make_column_transformer ((numerical_pipeline,numerical_features),
                                     (categorical_pipeline,categorical_fetures))
model= make_pipeline(preprocessor,SGDClassifier())
f=model.fit(X,y)
print(f)
#######
#pipelines parallélle plusieurs transformeurs
print("parallele make union")
numerical_features=X[['age','fare']]
pipeline=make_union(StandardScaler(),Binarizer())
print(pipeline.fit_transform(numerical_features))
print(pipeline.fit_transform(numerical_features).shape)