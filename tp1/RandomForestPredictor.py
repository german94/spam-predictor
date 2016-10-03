##########################################
#Importante:
#Para correr este archivo se necesitan tener los siguientes directorios y archivos:
#Directorio train con archivos df, ham_train_dev, spam_train_dev, word_count_att_names
#Directorio test con archivos spam_test_dev y ham_test_dev
#Directorio resultados con subdirectorio RandomForest, dentro de RandomForest tiene que estar el archivo best_rf_clf

# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

import time
import pickle
import ClfGenerator as cflgen
import pandas as pd
from sklearn.tree import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Intenta cargar el clasificador ya entrenado. Si no existe, lo entrena y guarda para futuros usos.
try:
  rf_clf = pickle.load(open("resultados/RandomForest/clasificadorEntrenado_400att", "r"))
except IOError:
  # Si no existe el dataframe, o la lista de palabras mas usadas, o el mejor decision tree
  # classifier surgido de grid search, es porque no se corrio ClfGenerator.py
  # (o hubo un error al correrlo)
  try:
    df = pickle.load(open("train/df", "r"))
    word_count_att_names = pickle.load(open("train/word_count_att_names", "r"))
    rf_clf = pickle.load(open("resultados/RandomForest/best_rf_clf", "r"))
  except IOError as e:
    print "Correr primero ClfGenerator.py"
    raise e

  word_count_att_df_names = map(lambda x : 'w_' + cflgen.deleteSpecialChars(x), word_count_att_names)
  X = df[['len', 'count_spaces', 'links', 'tags', 'rare'] + word_count_att_df_names].values
  y = df['class']

  # Aplicamos seleccion de atributos a la matriz X
  print "Seleccionando mejores 400 atributos..."
  startTimeAttSelection = time.time()
  best_attrs = SelectKBest(chi2, k=400)
  X_new = best_attrs.fit_transform(X, y)
  endTimeAttSelection = time.time()
  print "Tiempo consumido: " + str(endTimeAttSelection - startTimeAttSelection) 
  
  #Guardo los mejores atributos elegidos para luego poder hacer el transform sobre los mails a predecir
  file = open("resultados/RandomForest/best_attrs", "w")
  pickle.dump(best_attrs, file)
  file.close()

  print "Entrenando modelo..."
  startTimeTrain = time.time()
  rf_clf.fit(X_new, y)
  endTimeTrain = time.time()
  print "Tiempo consumido: " + str(endTimeTrain - startTimeTrain)

  #Guardo el clasificador entrenado
  file = open("resultados/RandomForest/clasificadorEntrenado_400att", "w")
  pickle.dump(rf_clf, file)
  file.close()

# En esta seccion se procesaran los mails a predecir.
# Si no se habia cargado ya la lista de atributos, la cargo
try:
  word_count_att_names
except NameError:
  # Si no la puedo cargar, es porque no existe y primero hay que correr ClfGenerator.py
  try:
    word_count_att_names = pickle.load(open("train/word_count_att_names", "r"))
  except IOError as e:
    print "Correr primero ClfGenerator.py"
    raise e

# Intento cargar el dataframe del dataset de test. Si no existe, lo creo.
try:
  df_test = pickle.load(open("test/df_test_TEST", "r"))
except IOError:
  ham_test = pickle.load(open("test/ham_test_dev", "r"))
  spam_test = pickle.load(open("test/spam_test_dev", "r"))

  # Armo un dataset de Pandas 
  # http://pandas.pydata.org/
  df_test = pd.DataFrame(ham_test+spam_test, columns=['text'])
  df_test['class'] = ['ham' for _ in range(len(ham_test))]+['spam' for _ in range(len(spam_test))]

  # Extraigo atributos simples: 
  # 1) Longitud del mail.
  df_test['len'] = map(len, df_test.text)
  # 2) Cantidad de espacios en el mail.
  df_test['count_spaces'] = map(cflgen.count_spaces, df_test.text)
  # 3) Cantidad de links
  df_test['links'] = map(cflgen.count_links, df_test.text)
  # 4) Cantidad de tags
  df_test['tags'] = map(cflgen.count_html_tags, df_test.text)
  # 5) cantidad de caracteres raros
  df_test['rare'] = map(cflgen.count_rare_chars, df_test.text)

  # Conseguimos los cuerpos de cada mail
  all_mail_bodies = ham_test + spam_test
  all_mail_bodies[:] = map((lambda x : cflgen.get_last_message(cflgen.get_mail_body(x))), all_mail_bodies)

  # Contamos la cantidad de apariciones de las palabras mas utilizadas del dataset,
  # y las agregamos al dataframe
  for word in word_count_att_names:
    df_test['w_' + cflgen.deleteSpecialChars(word)] = map(lambda x : x.count(word), all_mail_bodies)

  # Guardamos el dataframe de test
  file = open("test/df_test_TEST", "w")
  pickle.dump(df_test, file)
  file.close()

word_count_att_df_names = map(lambda x : 'w_' + cflgen.deleteSpecialChars(x), word_count_att_names)
X_test = df_test[['len', 'count_spaces', 'links', 'tags', 'rare'] + word_count_att_df_names].values
y_test = df_test['class']

# Si no habiamos cargado ya los mejores atributos, los levanto.
try:
  best_attrs
except NameError:
  best_attrs = pickle.load(open("resultados/RandomForest/best_attrs", "r"))

print "Aplicando seleccion de atributos y prediciendo..."
startTestTransformTime = time.time()
X_new_test = best_attrs.transform(X_test)
predArr = rf_clf.predict(X_new_test)
endTestTransformTime = time.time()

print "Tiempo consumido en la prediccion (transform + predict) " + str(endTestTransformTime - startTestTransformTime)
print "Accuracy de random forest sobre test: " + str(accuracy_score(predArr, y_test))
print "f1_score de random forest sobre test: " + str(f1_score(predArr, y_test, pos_label="spam"))
