# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

import json
import numpy as np
import pandas as pd
import re
import email
import string
import pickle
from functools import reduce
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

# Extraemos las palabras mas utilizadas en los cuerpos de los mails y guardamos
# como atributo la cantidad de apariciones de cada una.
word_count_att_names = pickle.load(open("train/word_count_att_names", "rb"))

#codigo para guardar word_count_att_names
df = pickle.load(open("train/df", "rb"))

X = df[['len', 'count_spaces', 'links', 'tags','rare'] + word_count_att_names].values
y = df['class']
#############################################
best_decision_tree_clf = pickle.load(open("resultados/DecisionTreeClass/best_decision_tree_clf", "rb"))
best_mnb_clf = pickle.load(open("resultados/naiveBayesMultinom/best_mnb_clf", "rb"))
best_rf_clf = pickle.load(open("resultados/RandomForest/best_rf_clf", "rb"))

#El siguiente bloque de codigo comentado se utilizo para realizar la extraccion de los veinte atributos mas descriptivos utilizando
#chi2 y f_classif, ambas con la funciion SelectKBest
# atributos = ['len', 'count_spaces', 'links', 'tags','rare'] + word_count_att_names
# best = SelectKBest(chi2, k=20)
# X_new = best.fit_transform(X, y)
# support = best.get_support()
# attr_usados = [atributos[i] for i in range(0, len(support)) if support[i]]
# print "20 MEJORES ATRIBUTOS CON CHI2"
# for k in attr_usados:
# 	print k

# best = SelectKBest(f_classif, k=20)
# X_new = best.fit_transform(X, y)
# support = best.get_support()
# attr_usados = [atributos[i] for i in range(0, len(support)) if support[i]]
# print "20 MEJORES ATRIBUTOS CON F_CLASSIF"
# for k in attr_usados:
# 	print k

#Los siguientes bloques de codigo los utilizamos para realizar reduccion de dimensionalidad y para seleccionar los k atributos
#mas descriptivos. En el caso de PCA no pudimos utilizar el clasificador multinomial por un tema de valores negativos que aparecian
#y no lo hacian compatible con la transformacion que realiza PCA.
#Ademas de la seleccion de atributos tambien experimentamos con la funcion SelectPercentile con varios percentiles.
classifiersPCA = [best_decision_tree_clf, best_rf_clf, GaussianNB()]
for clf in classifiersPCA:
	estimatorsOnlyPCA = [('reduce_dim', PCA()), ('clf', clf)]
	clfpip = Pipeline(estimatorsOnlyPCA)
	print "Ejecutando " + type(clf).__name__ + " con PCA"
	pca_res = cross_val_score(clfpip, X, y, cv=10)
	print np.mean(pca_res), np.std(pca_res)

print "_____________________________________________________"
print "K-Best selection"
number_of_attrs = [20, 50, 100, 150, 200, 250, 300, 320, 370, 400, 420, 450]
classifiersKBest = [best_decision_tree_clf, best_rf_clf, best_mnb_clf, GaussianNB()]
for n_attrs in number_of_attrs:
    for clf in classifiersKBest:
    	estimatorsOnlyKBestAtt = [('k_best', SelectKBest(chi2, k=n_attrs)), ('clf', clf)]
    	clfpip = Pipeline(estimatorsOnlyKBestAtt)
        print "Ejecutando " + type(clf).__name__ + " con mejores " + str(n_attrs) + " atributos..."
        best_attrs_res = cross_val_score(clfpip, X, y, cv=10)
        print np.mean(best_attrs_res), np.std(best_attrs_res)

percentiles = [5, 10, 20, 25]
classifiers = [best_decision_tree_clf, best_rf_clf, GaussianNB(), best_mnb_clf]
print "_____________________________________________________"
print "K-Best selection con f_classif"
for n_attrs in number_of_attrs:
    for clf in classifiers:
    	estimatorsOnlyKBestAtt = [('k_best', SelectKBest(f_classif, k=n_attrs)), ('clf', clf)]
    	clfpip = Pipeline(estimatorsOnlyKBestAtt)
        print "Ejecutando " + type(clf).__name__ + " con mejores " + str(n_attrs) + " atributos..."
        best_attrs_res = cross_val_score(clfpip, X, y, cv=10)
        print np.mean(best_attrs_res), np.std(best_attrs_res)


print "_____________________________________________________"
print "PCA + K-Best selection con f_classif"
for n_attrs in number_of_attrs:
    for clf in classifiers:
    	estimatorsPCAAndKbestAtt = [('reduce_dim', PCA()), ('k_best', SelectKBest(f_classif, k=n_attrs)), ('clf', clf)]
    	clfpip = Pipeline(estimatorsPCAAndKbestAtt)
        print "Ejecutando " + type(clf).__name__ + " con PCA y mejores " + str(n_attrs) + " atributos..."
        best_attrs_res = cross_val_score(clfpip, X, y, cv=10)
        print np.mean(best_attrs_res), np.std(best_attrs_res)

print "_____________________________________________________"
print "SelectPercentile con f_classif"
for clf in classifiers:
	for p in percentiles:
		estimatorsOnlyKBestAtt = [('select_perc', SelectPercentile(f_classif, percentile=p)), ('clf', clf)]
		clfpip = Pipeline(estimatorsOnlyKBestAtt)
		print "Ejecutando " + type(clf).__name__ + " con percentile = " + str(p)
		best_attrs_res = cross_val_score(clfpip, X, y, cv=10)
		print np.mean(best_attrs_res), np.std(best_attrs_res)	


print "_____________________________________________________"
print "PCA + SelectPercentile con f_classif"
for clf in classifiers:
	for p in percentiles:
		estimatorsPCAAndKbestAtt = [('reduce_dim', PCA()), ('select_perc', SelectPercentile(f_classif, percentile=p)), ('clf', clf)]
		clfpip = Pipeline(estimatorsPCAAndKbestAtt)
		print "Ejecutando " + type(clf).__name__ + " con PCA y percentile = " + str(p)
		best_attrs_res = cross_val_score(clfpip, X, y, cv=10)
		print np.mean(best_attrs_res), np.std(best_attrs_res)