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

# Leo los mails.
ham_txt = json.load(open(dataset_path + 'ham_dev.json'))[]
spam_txt = json.load(open(dataset_path + 'spam_dev.json'))[]

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

# Creamos un decision tree classifier
decision_tree_clf = DecisionTreeClassifier()

# Escribimos todas los parametros que nos gustaria variar en el decision tree
decision_tree_param_grid = {
"criterion": ["gini", "entropy"],
"max_features": [1, 3, 7, 10],
"max_depth": [2, 3, 5, None],
"min_samples_split": [1, 3, 5, 10]}

# Corremos un grid search para ver que combinacion de atributos es la mejor
grid_search = GridSearchCV(decision_tree_clf, param_grid=decision_tree_param_grid)
grid_search.fit(X, y)

print "Mejor puntaje de decision tree despues de correr grid search: " + str(grid_search.best_score_)

# Creamos un nuevo decision tree a partir de la mejor combinacion de atributos
# dada por el grid search
decision_tree_best_params = grid_search.best_params_
best_decision_tree_clf = DecisionTreeClassifier(
    criterion=decision_tree_best_params['criterion'],
    max_depth=decision_tree_best_params['max_depth'],
    max_features=decision_tree_best_params['max_features'],
    min_samples_split=decision_tree_best_params['min_samples_split']
    )

# Ejecuto el clasificador entrenando con un esquema de cross validation
# de 10 folds.
print "Ejecutando clasificador"
res = cross_val_score(clf, X, y, cv=10)
print np.mean(res), np.std(res)
# salida: 0.687566666667 0.0190878702354  (o similar)

