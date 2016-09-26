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
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA

# Extraemos las palabras mas utilizadas en los cuerpos de los mails y guardamos
# como atributo la cantidad de apariciones de cada una.
word_count_att_names = pickle.load(open("word_count_att_names", "rb"))

#codigo para guardar word_count_att_names
df = pickle.load(open("resultados/df", "rb"))

#Las siguientes dos lineas son para levantar el df y el word_count_att_names de archivos, por eso estan comentadas
#df = pickle.load(open("df", "r"))
#word_count_att_names = pickle.load(open("word_count_att_names", "r"))
# Preparo data para clasificar
X = df[['len', 'count_spaces', 'links', 'tags','rare'] + word_count_att_names].values
y = df['class']
#############################################

print "Aplicando reduccion de dimensionalidad/transformacion de datos"

# PCA
print "Aplicando PCA"
# X_PCA = PCA().fit_transform(X, y)
# file = open("resultados/X_PCA", "wb")
# pickle.dump(X_PCA, file)
# file.close()


best_decision_tree_clf = pickle.load(open("resultados/DecisionTreeClass/best_decision_tree_clf", "rb"))
best_mnb_clf = pickle.load(open("resultados/naiveBayesMultinom/best_mnb_clf", "rb"))
best_rf_clf = pickle.load(open("resultados/RandomForest/best_rf_clf", "rb"))

#classifiersPCA = [best_decision_tree_clf, best_rf_clf, GaussianNB()]

# for clf in classifiersPCA:
#     print "Ejecutando " + type(clf).__name__ + " con PCA"
#     pca_res = cross_val_score(clf, X_PCA, y, cv=10)
#     print np.mean(pca_res), np.std(pca_res)


classifiers = [best_decision_tree_clf, best_rf_clf, best_mnb_clf, GaussianNB()]

# Feature selection usando KBest. Usamos chi2 porque estamos usando estructuras
# esparsas, y la unica funcion que trabaja con ellas sin volverlas densas es chi2.
# Probamos que cantidad de atributos es mejor, variando el valor entre 20, 50, 100 y 250
print "Aplicando K-Best selection"
number_of_attrs = [300, 320, 370, 400, 420, 450]

for n_attrs in number_of_attrs:
    X_new = SelectKBest(chi2, k=n_attrs).fit_transform(X, y)
    file = open("resultados/X_new_" + str(n_attrs), "wb")
    pickle.dump(X_new, file)
    file.close()
    for clf in classifiers:
        print "Ejecutando " + type(clf).__name__ + " con mejores " + str(n_attrs) + " atributos..."
        best_attrs_res = cross_val_score(clf, X_new, y, cv=10)
        print np.mean(best_attrs_res), np.std(best_attrs_res)