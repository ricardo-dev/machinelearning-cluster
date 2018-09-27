#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:39:20 2018

@author: ricardo
"""

## kmeans - machine learning sem supervisao para clustering (grupo)

## base de dados - analise de celebridades
## atributos:
# nome
# Op - abertura mental para novas experiencias
# Co - Organizacao 
# Ex - Grau de timidez
# Ag - grau de empatia com os outros
# Ne - Grau de irritabilidade
# Wordcount - numero medio de palavras no tweets
# Categoia - Atividade de trabalho
# 1 Ator/Atriz,2 cantor,3 modelo,4 tv/serie,5 radio,6 tecnologia
# 7 esportes, 8 politica, 9 escritor

## Neste exemplo sera usado 3 campos
## classificacao -> por semelhancas de personalidade

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# grafico
from mpl_toolkits.mplot3d import Axes3D # 3D
plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('ggplot')

dataframe = pd.read_csv('analisis.csv')
#print(dataframe.head())
#print(dataframe.describe()) # dados estatisticos
print( dataframe.groupby('categoria').size() ) 

dataframe.drop( ['categoria'], 1 ).hist()
plt.show()

## Selecionando 3 campos para cruzar os dados
## e tentar descobrir do cluster e relacao das categorias

sb.pairplot( dataframe.dropna() , hue='categoria', size=4, vars=["op","ex","ag"], kind='scatter' )
## nao e visivel nenhum agrupamento

X = np.array( dataframe[["op","ex","ag"]] )
y = np.array( dataframe['categoria'] )
X.shape
fig = plt.figure()
ax = Axes3D(fig)
cores = ['blue','red','green','blue','cyan','yellow','orange','black','pink','brown','purple']
asgnar = []
for row in y:
    asgnar.append( cores[row] )
ax.scatter( X[:,0], X[:,1], X[:,2], c=asgnar, s=60)

### metodo de Elbow
#WCSS = []
#print('Valores de K')
#for i in range(1,11):
#    kmeans = KMeans(n_clusters = i, init='random')
#    kmeans.fit(X)
#    print(i, kmeans.inertia_)
#    WCSS.append(kmeans.inertia_)
#fig = plt.figure()
#plt.plot(range(1,11), WCSS)
#plt.title('Metodo Elbow')
#plt.xlabel('Numero de clusters')
#plt.ylabel('WSS')
#plt.show()

## neste estudo, n = 5 ficou ideal

kmeans = KMeans(n_clusters=5, init='random').fit(X)
centroid = kmeans.cluster_centers_
print(centroid)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_
cores = ['red', 'green', 'blue', 'cyan', 'yellow']
asgnar = []
for row in labels:
    asgnar.append(cores[row])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:,0], X[:,1], X[:, 2], c=asgnar, s=60)
ax.scatter(C[:,0], C[:,1], C[:, 2], marker = '*', c=cores, s=1000)

## Estudando os resultados
## projecao 2D
## 1
fig = plt.figure()
f1 = dataframe['op'].values
f2 = dataframe['ex'].values

plt.scatter( f1, f2, c=asgnar, s=70 )
plt.scatter( C[:,0], C[:,1], marker="*", c=cores, s=1000 )
plt.show()
## 2
fig = plt.figure()
f1 = dataframe['op'].values
f2 = dataframe['ag'].values

plt.scatter( f1, f2, c=asgnar, s=70 )
plt.scatter( C[:,0], C[:,2], marker="*", c=cores, s=1000 )
plt.show()
## 3
fig = plt.figure()
f1 = dataframe['ag'].values
f2 = dataframe['ex'].values

plt.scatter( f1, f2, c=asgnar, s=70 )
plt.scatter( C[:,2], C[:,1], marker="*", c=cores, s=1000 )
plt.show()

## analisando os resultados
copy = pd.DataFrame()
copy['usuario'] = dataframe['usuario'].values
copy['categoria'] = dataframe['categoria'].values
copy['label'] = labels
quantidadeGrupo = pd.DataFrame()
quantidadeGrupo['color'] = cores
quantidadeGrupo['quantidade'] = copy.groupby('label').size()
print(quantidadeGrupo)

group_referrer_index = copy['label'] == 2
group_referrals = copy[group_referrer_index]
diversidadeGrupo = pd.DataFrame()
diversidadeGrupo['categoria'] = [0,1,2,3,4,5,6,7,8,9]
diversidadeGrupo['quantidade'] = group_referrals.groupby('categoria').size()
print(diversidadeGrupo)

nomeGrupo = pd.DataFrame()
nomeGrupo['grupo'] = kmeans.labels_ 
nomeGrupo['nomes'] = dataframe['usuario'].values
print(nomeGrupo)
    

## representante de cada cluster
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
users = dataframe['usuario'].values
for row in closest:
    print( users[row] )

X_new = np.array([ [45.92, 57.74, 16.66] ]) # novo registro
new_labels = kmeans.predict(X_new)
print(new_labels)
