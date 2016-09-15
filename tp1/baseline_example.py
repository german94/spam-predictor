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

# Leo los mails (poner los paths correctos).
ham_txt= json.load(open('ham_dev.json'))
spam_txt= json.load(open('spam_dev.json'))

def getMailBody(mail):
	chunks = ((re.split("\r\n\r\n|\n\n|\r\r", mail))[1:])
	if chunks == []:
		return ""
	return reduce( (lambda x,y : x + " " + y), chunks )

def getLastMessage(body):
	chunks = re.split("-----Original Message-----", body)
	return re.split("####", chunks[0])[0]

all_mails = ham_txt + spam_txt
map((lambda x : getLastMessage(getMailBody(x))), all_mails)


#print "------------------------------------------------------"
# print spam_txt[0]
#print "------------------------------------------------------"

# Armo un dataset de Pandas 
# http://pandas.pydata.org/

# df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
# df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

# Extraigo dos atributos simples: 
# 1) Longitud del mail.
# df['len'] = map(len, df.text)

# 2) Cantidad de espacios en el mail.
# def count_spaces(txt): return txt.count(" ")
# df['count_spaces'] = map(count_spaces, df.text)

# Preparo data para clasificar
# X = df[['len', 'count_spaces']].values
# y = df['class']

count_vectorizer = CountVectorizer(token_pattern='[^\d\W_]\w+', max_features=500)
word_count_matrix = count_vectorizer.fit_transform(all_mails)
word_count_matrix_t = word_count_matrix.transpose()
#print "Ham: "
word_count_att_names = count_vectorizer.get_feature_names()[:200]
print word_count_att_names

for att_name_idx in xrange(len(word_count_att_names)):
	df[word_count_att_names[att_name_idx]] = word_count_matrix_t[att_name_idx]

#vectorizerSpam = CountVectorizer(token_pattern='[^\d\W_]\w+', max_features=500)
#spam_word_count_matrix = vectorizerSpam.fit_transform(spam_txt)
#print "Spam: "
#print vectorizerSpam.get_feature_names()[200:]




# Elijo mi clasificador.
#clf = DecisionTreeClassifier()

# Ejecuto el clasificador entrenando con un esquema de cross validation
# de 10 folds.
#res = cross_val_score(clf, X, y, cv=10)
#print np.mean(res), np.std(res)
# salida: 0.687566666667 0.0190878702354  (o similar)