##########################################
#Importante:
#Para correr este archivo se necesitan tener los siguientes directorios y archivos:
#Directorio train con archivos df, ham_train_dev, spam_train_dev, word_count_att_names
#Directorio test con archivos spam_test_dev y ham_test_dev
#Directorio resultados con subdirectorio RandomForest, dentro de RandomForest tiene que estar el archivo best_rf_clf

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
from sklearn.decomposition import PCA

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

def raros(s): 
    cant = 0
    for x in s:
        if (1 <= ord(x) and ord(x) <= 32) or (127 <= ord(x) and ord(x) <= 160):
            cant = cant +1
    return cant     

def deleteSpecialChars(s):
    return re.sub('[^A-Za-z0-9]+', '', s)

#DATOS DE TRAIN: en esta seccion se deben entrenar los modelos si no se cargan ya entrenados
df = pickle.load(open("train/df", "rb"))
word_count_att_names = pickle.load(open("train/word_count_att_names", "rb"))
X = df[['len', 'count_spaces', 'links', 'tags','rare'] + word_count_att_names].values
y = df['class']
rf_clf = pickle.load(open("resultados/RandomForest/best_rf_clf", "rb"))

startTimeAttSelection = time.time()
best = SelectKBest(chi2, k=250) 
X_new = best.fit_transform(X, y)
endTimeAttSelection = time.time()
print "Tiempo que se tardo en seleccionar los 250 mejores atributos y generar la matriz: " + str(endTimeAttSelection - startTimeAttSelection) 
#Guardo el best para luego poder hacer el transform sobre los mails a predecir
file = open("resultados/RandomForest/best", "wb")
pickle.dump(best, file)
file.close()
print "Ejecutando Random Forest con los mejores 250 atributos..."
best_attrs_res = cross_val_score(rf_clf, X_new, y, cv=10)
print np.mean(best_attrs_res), np.std(best_attrs_res)

startTimeTrain = time.time()
rf_clf.fit(X_new, y)
endTimeTrain = time.time()
print "Tiempo que se tardo en entrenar: " + str(endTimeTrain - startTimeTrain)
#Guardo el clasificador entrenado
file = open("resultados/RandomForest/clasificadorEntrenado_250att", "wb")
pickle.dump(rf_clf, file)
file.close()





#DATOS DE TESET: en esta seccion se deben usar los modelos entrenados para hacer el predict
ham_test = pickle.load(open("test/ham_test_dev", "rb"))
spam_test = pickle.load(open("test/spam_test_dev", "rb"))

# Armo un dataset de Pandas 
# http://pandas.pydata.org/
df_test = pd.DataFrame(ham_test+spam_test, columns=['text'])
df_test['class'] = ['ham' for _ in range(len(ham_test))]+['spam' for _ in range(len(spam_test))]

# Extraigo atributos simples: 
# 1) Longitud del mail.
df_test['len'] = map(len, df_test.text)
# 2) Cantidad de espacios en el mail.
df_test['count_spaces'] = map(count_spaces, df_test.text)
# 3) Cantidad de links
df_test['links'] = map(getLinks, df_test.text)
# 4) Cantidad de tags
df_test['tags'] = map(get_tags_regex, df_test.text)
# 5) cantidad de caracteres raros
df_test['rare'] = map(raros,df_test.text)

# Conseguimos los cuerpos de cada mail
all_mail_bodies = ham_test + spam_test
all_mail_bodies[:] = map((lambda x : getLastMessage(getMailBody(x))), all_mail_bodies)

# Extraemos las palabras mas utilizadas en los cuerpos de los mails y guardamos
# como atributo la cantidad de apariciones de cada una.
print "Extrayendo palabras mas usadas"
count_vectorizer = CountVectorizer(token_pattern='[^\d\W_]\w+', max_features=500)
word_count_matrix_test = count_vectorizer.fit_transform(all_mail_bodies)
word_count_matrix_test = word_count_matrix_test.transpose().toarray()
word_count_att_names_test = count_vectorizer.get_feature_names()
word_count_att_names_test[:] = map(lambda x : 'w' + deleteSpecialChars(x), word_count_att_names_test)

#codigo para guardar word_count_att_names
file = open("test/word_count_att_names_TEST", "wb")
pickle.dump(word_count_att_names_test, file)
file.close()

print "Agregando palabras mas usadas como atributos"
for att_name_idx in xrange(len(word_count_att_names_test)):
	df_test[word_count_att_names_test[att_name_idx]] = word_count_matrix_test[att_name_idx]

#codigo para guardar df_test
file = open("test/df_test_TEST", "wb")
pickle.dump(df_test, file)
file.close()

X_test = df_test[['len', 'count_spaces', 'links', 'tags','rare'] + word_count_att_names_test].values
y_test = df_test['class']

startTestTransformTime = time.time()
X_new_test = best.transform(X_test)
predArr = rf_clf.predict(X_new_test)
endTestTransformTime = time.time()
print "Tiempo que tardo en predecir (transform + predict) " + str(endTestTransformTime - startTestTransformTime)
print "Accuracy de random forest sobre test: " + str(accuracy_score(predArr, y_test))
print "f1_score de random forest sobre test: " + str(f1_score(predArr, y_test, pos_label="spam"))