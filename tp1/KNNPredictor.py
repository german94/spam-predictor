##########################################
#Importante:
#Para correr este archivo se necesitan tener los siguientes directorios y archivos:
#Directorio train con archivos df, ham_train_dev, spam_train_dev, word_count_att_names
#Directorio test con archivos spam_test_dev y ham_test_dev
#Directorio resultados con subdirectorio KNN para guardar algunas cosas

#Al correr esto 

# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

import json
import numpy as np
import pandas as pd
import re
import email
import string
import pickle
import time
from functools import reduce
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
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
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

#DATOS DE TRAIN: en esta seccion se deben entrenar los modelos si no se cargan ya entrenados
df = pickle.load(open("train/df", "rb"))
word_count_att_names = pickle.load(open("train/word_count_att_names", "rb"))
X = df[['len', 'count_spaces', 'links', 'tags','rare'] + word_count_att_names].values
y = df['class']
knn_clf = KNeighborsClassifier(
 n_neighbors = 3,
weights = "distance",
n_jobs = -1
)

X_PCA = PCA().fit_transform(X, y)

startTimeAttSelection = time.time()
best = SelectKBest(f_classif, k=50)
X_new = best.fit_transform(X_PCA, y)
endTimeAttSelection = time.time()
print "Tiempo que se tardo en seleccionar los 50 mejores atributos y generar la matriz: " + str(endTimeAttSelection - startTimeAttSelection) 
#Guardo el best para luego poder hacer el transform sobre los mails a predecir
file = open("resultados/KNN/best", "wb")
pickle.dump(best, file)
file.close()
print "Ejecutando KNN con los mejores 50 atributos..."
best_attrs_res = cross_val_score(knn_clf, X_new, y, cv=10)
print np.mean(best_attrs_res), np.std(best_attrs_res)

startTimeTrain = time.time()
knn_clf.fit(X_new, y)
endTimeTrain = time.time()
print "Tiempo que se tardo en entrenar: " + str(endTimeTrain - startTimeTrain)
#Guardo el clasificador entrenado
file = open("resultados/KNN/clasificadorEntrenado_50att", "wb")
pickle.dump(knn_clf, file)
file.close()




# Todo el siguiente bloque de codigo esta comentado porque ya se utilizo una vez y ahora solo basta con cargar los datos desde los archivos
# #DATOS DE TESET: en esta seccion se deben usar los modelos entrenados para hacer el predict
# ham_test = pickle.load(open("test/ham_test_dev", "rb"))
# spam_test = pickle.load(open("test/spam_test_dev", "rb"))

# # Armo un dataset de Pandas 
# # http://pandas.pydata.org/
# df_test = pd.DataFrame(ham_test+spam_test, columns=['text'])
# df_test['class'] = ['ham' for _ in range(len(ham_test))]+['spam' for _ in range(len(spam_test))]

# # Extraigo atributos simples: 
# # 1) Longitud del mail.
# df_test['len'] = map(len, df_test.text)
# # 2) Cantidad de espacios en el mail.
# df_test['count_spaces'] = map(count_spaces, df_test.text)
# # 3) Cantidad de links
# df_test['links'] = map(getLinks, df_test.text)
# # 4) Cantidad de tags
# df_test['tags'] = map(get_tags_regex, df_test.text)
# # 5) cantidad de caracteres raros
# df_test['rare'] = map(raros,df_test.text)

# # Conseguimos los cuerpos de cada mail
# all_mail_bodies = ham_test + spam_test
# all_mail_bodies[:] = map((lambda x : getLastMessage(getMailBody(x))), all_mail_bodies)

# # Extraemos las palabras mas utilizadas en los cuerpos de los mails y guardamos
# # como atributo la cantidad de apariciones de cada una.
# print "Extrayendo palabras mas usadas"
# count_vectorizer = CountVectorizer(token_pattern='[^\d\W_]\w+', max_features=500)
# word_count_matrix_test = count_vectorizer.fit_transform(all_mail_bodies)
# word_count_matrix_test = word_count_matrix_test.transpose().toarray()
# word_count_att_names_test = count_vectorizer.get_feature_names()
# word_count_att_names_test[:] = map(lambda x : 'w' + deleteSpecialChars(x), word_count_att_names_test)

#codigo para guardar word_count_att_names
# file = open("test/word_count_att_names_TEST", "wb")
# pickle.dump(word_count_att_names_test, file)
# file.close()

# print "Agregando palabras mas usadas como atributos"
# for att_name_idx in xrange(len(word_count_att_names_test)):
# 	df_test[word_count_att_names_test[att_name_idx]] = word_count_matrix_test[att_name_idx]

# #codigo para guardar df_test
# file = open("test/df_test_TEST", "wb")
# pickle.dump(df_test, file)
# file.close()

df_test = pickle.load(open("test/df_test_TEST", "rb"))
word_count_att_names_test = pickle.load(open("test/word_count_att_names_TEST", "rb"))


X_test = df_test[['len', 'count_spaces', 'links', 'tags','rare'] + word_count_att_names_test].values
y_test = df_test['class']

startTestTransformTime = time.time()
X_new_test = best.transform(PCA().transform(X_test))
predArr = knn_clf.predict(X_new_test)
endTestTransformTime = time.time()
print "Tiempo que tardo en predecir (transform + predict) " + str(endTestTransformTime - startTestTransformTime)
print "Accuracy de KNN sobre test: " + str(accuracy_score(predArr, y_test))
print "f1_score de KNN sobre test: " + str(f1_score(predArr, y_test, pos_label="spam"))