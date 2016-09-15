# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

import json
import numpy as np
import pandas as pd
import re
import email
import string
from functools import reduce
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

dataset_path = 'dataset_dev/'

def getMailBody(mail):
	chunks = ((re.split("\r\n\r\n|\n\n|\r\r", mail))[1:])
	if chunks == []:
		return ""
	return reduce( (lambda x,y : x + " " + y), chunks )

def getLastMessage(body):
	chunks = re.split("-----Original Message-----", body)
	return re.split("####", chunks[0])[0]

#print "------------------------------------------------------"
# print spam_txt[0]
#print "------------------------------------------------------"

# Leo los mails.
ham_txt = json.load(open(dataset_path + 'ham_dev.json'))
spam_txt = json.load(open(dataset_path + 'spam_dev.json'))

# Armo un dataset de Pandas 
# http://pandas.pydata.org/
df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

# Extraigo dos atributos simples: 
# 1) Longitud del mail.
df['len'] = map(len, df.text)

# 2) Cantidad de espacios en el mail.
def count_spaces(txt): return txt.count(" ")
df['count_spaces'] = map(count_spaces, df.text)

# Conseguimos los cuerpos de cada mail
all_mail_bodies = ham_txt + spam_txt
map((lambda x : getLastMessage(getMailBody(x))), all_mail_bodies)

# Extraemos las palabras mas utilizadas en los cuerpos de los mails y guardamos
# como atributo la cantidad de apariciones de cada una.
print "Extrayendo palabras mas usadas"
count_vectorizer = CountVectorizer(token_pattern='[^\d\W_]\w+', max_features=500)
word_count_matrix = count_vectorizer.fit_transform(all_mail_bodies)
word_count_matrix_t = word_count_matrix.transpose().toarray()
word_count_att_names = count_vectorizer.get_feature_names()
print "Agregando palabras mas usadas como atributos"
for att_name_idx in xrange(len(word_count_att_names)):
	df[word_count_att_names[att_name_idx]] = word_count_matrix_t[att_name_idx]

# Preparo data para clasificar
X = df[['len', 'count_spaces'] + word_count_att_names].values
y = df['class']

# Elijo mi clasificador.
clf = DecisionTreeClassifier()

# Ejecuto el clasificador entrenando con un esquema de cross validation
# de 10 folds.
print "Ejecutando clasificador"
res = cross_val_score(clf, X, y, cv=10)
print np.mean(res), np.std(res)
# salida: 0.687566666667 0.0190878702354  (o similar)