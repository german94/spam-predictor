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

#esta funcion es la que limpia de tags html a un texto y ademas te devuelve los tags (start tags, se puede devolver tambien los endtags pero me parecio medio al pedo)
def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return (s.get_data(), s.tags)

def count_spaces(txt): 
    return txt.count(" ")

dataset_path = ''

def getMailBody(mail):
	chunks = ((re.split("\r\n\r\n|\n\n|\r\r", mail))[1:])
	if chunks == []:
		return ""
	return reduce( (lambda x,y : x + " " + y), chunks )

def getLastMessage(body):
	chunks = re.split("-----Original Message-----", body)
	return re.split("####", chunks[0])[0]

def getLinks(text):
	return len(re.findall("(?P<url>https?://[^\s]+)", text))

def get_tags_regex(html):
	return len(re.findall('<[^<]+?>', html))

def clean_tags_regex(html):
	return re.sub('<[^<]+?>', '', html)

#print "------------------------------------------------------"
# print spam_txt[0]
#print "------------------------------------------------------"

#Esto ahora ya no hace falta porque leemos las listas correspondientes a los mails de spam y ham para train y los mails de test
# Leo los mails.
#ham_txt = json.load(open(dataset_path + 'ham_dev.json'))[]
#spam_txt = json.load(open(dataset_path + 'spam_dev.json'))[]

#Esto simplemente levanta las listas de los mails desde los archivos
ham_txt = pickle.load(open("ham_train_dev", "rb"))
spam_txt = pickle.load(open("spam_train_dev", "rb"))
test_txt = pickle.load(open("test_dev", "rb"))

# Armo un dataset de Pandas 
# http://pandas.pydata.org/
df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

# Extraigo atributos simples: 
# 1) Longitud del mail.
df['len'] = map(len, df.text)
# 2) Cantidad de espacios en el mail.
df['count_spaces'] = map(count_spaces, df.text)
# 3) Cantidad de links
df['links'] = map(getLinks, df.text)
# 4) Cantidad de tags
df['tags'] = map(get_tags_regex, df.text)

# Conseguimos los cuerpos de cada mail
all_mail_bodies = ham_txt + spam_txt
all_mail_bodies[:] = map((lambda x : getLastMessage(getMailBody(x))), all_mail_bodies)

# Extraemos las palabras mas utilizadas en los cuerpos de los mails y guardamos
# como atributo la cantidad de apariciones de cada una.
print "Extrayendo palabras mas usadas"
count_vectorizer = CountVectorizer(token_pattern='[^\d\W_]\w+', max_features=500)
word_count_matrix = count_vectorizer.fit_transform(all_mail_bodies)
word_count_matrix_t = word_count_matrix.transpose().toarray()
word_count_att_names = count_vectorizer.get_feature_names()
word_count_att_names[:] = map(lambda x : 'w' + x, word_count_att_names)

#codigo para guardar word_count_att_names
file = open("word_count_att_names", "wb")
pickle.dump(word_count_att_names, file)
file.close()

print "Agregando palabras mas usadas como atributos"
for att_name_idx in xrange(len(word_count_att_names)):
	df[word_count_att_names[att_name_idx]] = word_count_matrix_t[att_name_idx]

#codigo para guardar df
file = open("df", "wb")
pickle.dump(df, file)
file.close()
#Las siguientes dos lineas son para levantar el df y el word_count_att_names de archivos, por eso estan comentadas
#df = pickle.load(open("df", "r"))
#word_count_att_names = pickle.load(open("word_count_att_names", "r"))
# Preparo data para clasificar
X = df[['len', 'count_spaces', 'links', 'tags'] + word_count_att_names].values
y = df['class']

#############################################

# Creamos un decision tree classifier
decision_tree_clf = DecisionTreeClassifier()

# Escribimos todas los parametros que nos gustaria variar en el decision tree
decision_tree_param_grid = {
    "criterion": ["gini", "entropy"],
    "max_features": [1, 3, 7, 10],
    "max_depth": [2, 3, 5, None],
    "min_samples_split": [1, 3, 5, 10]
}

# Corremos un grid search para ver que combinacion de atributos es la mejor
print "Corriendo grid search para los hiperparametros de decision tree..."
dt_grid_search = GridSearchCV(decision_tree_clf, param_grid=decision_tree_param_grid)
dt_grid_search.fit(X, y)

print "Mejor puntaje de decision tree despues de correr grid search: " + str(dt_grid_search.best_score_)

# Creamos un nuevo decision tree a partir de la mejor combinacion de atributos
# dada por el grid search
decision_tree_best_params = dt_grid_search.best_params_
best_decision_tree_clf = DecisionTreeClassifier(
    criterion=decision_tree_best_params['criterion'],
    max_depth=decision_tree_best_params['max_depth'],
    max_features=decision_tree_best_params['max_features'],
    min_samples_split=decision_tree_best_params['min_samples_split']
    )

# Ejecutamos el clasificador entrenando con un esquema de CV de 10 folds.
print "Ejecutando decision tree classifier"
dt_res = cross_val_score(best_decision_tree_clf, X, y, cv=10)
file = open("best_decision_tree_clf", "wb")
pickle.dump(best_decision_tree_clf, file)
file.close()
print np.mean(dt_res), np.std(dt_res)

#############################################

# Creamos un naive bayes multinomial
multinomial_nb_clf = MultinomialNB()

# Escribimos todos los parametros que nos gustaria variar
multinomial_nb_param_grid = {
    "alpha": [0.25, 0.5, 0.75, 1.0],
    "fit_prior": [True, False]
}

# Corremos un grid search para ver que combinacion de atributos es la mejor
print "Corriendo grid search para los hiperparametros de multinomial NB..."
mnb_grid_search = GridSearchCV(multinomial_nb_clf, param_grid=multinomial_nb_param_grid)
mnb_grid_search.fit(X, y)

print "Mejor puntaje de multinomial NB despues de correr grid search: " + str(mnb_grid_search.best_score_)

# Creamos un nuevo multinomial NB a partir de la mejor combinacion de atributos
# dada por el grid search
mnb_best_params = mnb_grid_search.best_params_
best_mnb_clf = MultinomialNB(
    alpha=mnb_best_params['params'],
    fit_prior=mnb_best_params['fit_prior']
    )

# Ejecutamos el clasificador entrenando con un esquema de CV de 10 folds.
print "Ejecutando multinomial NB"
mnb_res = cross_val_score(best_mnb_clf, X, y, cv=10)
file = open("best_mnb_clf", "wb")
pickle.dump(best_mnb_clf, file)
file.close()
print np.mean(mnb_res), np.std(mnb_res)

#############################################

# Creamos un naive bayes gaussiano
gaussian_nb_clf = GaussianNB()

# Ejecutamos el clasificador entrenando con un esquema de CV de 10 folds.
print "Ejecutando gaussian NB"
gaussian_nb_res = cross_val_score(gaussian_nb_clf, X, y, cv=10)
print np.mean(gaussian_nb_res), np.std(gaussian_nb_res)

#############################################

# Creamos un KNN classifier
knn_clf = KNeighborsClassifier()

# Escribimos todos los parametros que nos gustaria variar
knn_param_grid = {
    "n_neighbors": [3, 5, 10],
    "weights": ["uniform", "distance"],
    "algorithm": ["brute"], # como la matriz de count_vectorizer es sparse, hay que usar fuerza bruta
    "n_jobs": [-1] # esto es para que paralelice por cantidad de nucleos
}

# Corremos un grid search para ver que combinacion de atributos es la mejor
print "Corriendo grid search para los hiperparametros de KNN..."
knn_grid_search = GridSearchCV(knn_clf, param_grid=knn_param_grid)
knn_grid_search.fit(X, y)

print "Mejor puntaje de KNN despues de correr grid search: " + str(knn_grid_search.best_score_)

# Creamos un nuevo clasificador KNN a partir de la mejor combinacion de atributos
# dada por el grid search
knn_best_params = knn_grid_search.best_params_
best_knn_clf = KNeighborsClassifier(
    n_neighbors=knn_best_params['n_neighbors'],
    weights=knn_best_params['weights'],
    algorithm="brute",
    n_jobs=-1
    )

# Ejecutamos el clasificador entrenando con un esquema de CV de 10 folds.
print "Ejecutando KNN"
knn_res = cross_val_score(best_knn_clf, X, y, cv=10)
file = open("best_knn_clf", "wb")
pickle.dump(best_knn_clf, file)
file.close()
print np.mean(knn_res), np.std(knn_res)

#############################################

# Creamos un random forest classifier
rf_clf = RandomForestClassifier()

# Escribimos todos los parametros que nos gustaria variar
rf_param_grid = {
    "n_estimators": [5, 10, 20],
    "criterion": ["gini", "entropy"],
    "max_features": [1, 3, 7, 10],
    "max_depth": [2, 3, 5, None],
    "min_samples_split": [1, 3, 5, 10],
    "n_jobs": [-1] # esto es para que paralelice por cantidad de nucleos
}

# Corremos un grid search para ver que combinacion de atributos es la mejor
print "Corriendo grid search para los hiperparametros de random forest..."
rf_grid_search = GridSearchCV(rf_clf, param_grid=rf_param_grid)
rf_grid_search.fit(X, y)

print "Mejor puntaje de random forest despues de correr grid search: " + str(rf_grid_search.best_score_)

# Creamos un nuevo clasificador random forest a partir de la mejor combinacion de atributos
# dada por el grid search
rf_best_params = rf_grid_search.best_params_
best_rf_clf = RandomForestClassifier(
    n_estimators=rf_best_params['n_estimators'],
    criterion=rf_best_params['criterion'],
    max_features=rf_best_params['max_features'],
    max_depth=rf_best_params['max_depth'],
    min_samples_split=rf_best_params['min_samples_split'],
    n_jobs=-1
    )

# Ejecutamos el clasificador entrenando con un esquema de CV de 10 folds.
print "Ejecutando random forest"
rf_res = cross_val_score(best_rf_clf, X, y, cv=10)
file = open("best_rf_clf", "wb")
pickle.dump(best_rf_clf, file)
file.close()
print np.mean(rf_res), np.std(rf_res)

#############################################

# Creamos un SVM classifier
svm_clf = SVC()

# Escribimos todos los parametros que nos gustaria variar
svm_param_grid = {
    "kernel": ["poly", "rbf"],
    "degree": [3, 4, 5],
}

# Corremos un grid search para ver que combinacion de atributos es la mejor
print "Corriendo grid search para los hiperparametros de SVM..."
svm_grid_search = GridSearchCV(svm_clf, param_grid=svm_param_grid)
svm_grid_search.fit(X, y)

print "Mejor puntaje de SVM despues de correr grid search: " + str(svm_grid_search.best_score_)

# Creamos un nuevo clasificador SVM a partir de la mejor combinacion de atributos
# dada por el grid search
svm_best_params = svm_grid_search.best_params_
best_svm_clf = SVC(
    kernel=svm_best_params['kernel'],
    degree=svm_best_params['degree']
    )

# Ejecutamos el clasificador entrenando con un esquema de CV de 10 folds.
print "Ejecutando random forest"
svm_res = cross_val_score(best_svm_clf, X, y, cv=10)
print np.mean(svm_res), np.std(svm_res)


