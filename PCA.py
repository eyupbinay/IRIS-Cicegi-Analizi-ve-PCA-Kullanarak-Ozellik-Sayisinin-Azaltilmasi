#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "pca_iris.data"
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
df

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = df[features]
y = df[['target']]

x = StandardScaler().fit_transform(x)
x

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf

final_dataframe = pd.concat([principalDf, df[['target']]], axis = 1)
final_dataframe.head()

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['g', 'b', 'r']

plt.xlabel('principal component 1')
plt.ylabel('principal component 2')

for target, col in zip(targets,colors):
    dftemp = final_dataframe[df.target==target]
    plt.scatter(dftemp['principal component 1'], dftemp['principal component 2'], color=col)

pca.explained_variance_ratio_
pca.explained_variance_ratio_.sum()

