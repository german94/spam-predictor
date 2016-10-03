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

#################### Funciones auxiliares ####################

# Cuenta espacios en el mail
def count_spaces(txt): 
    return txt.count(" ")

# Se queda con el cuerpo de los mails (eliminando los headers).
# Para esto, chequea que haya dos (o mas) lineas nuevas de texto seguidas
# (ya que el header no contiene dos lineas nuevas seguidas)
def get_mail_body(mail):
	chunks = ((re.split("\r\n\r\n|\n\n|\r\r", mail))[1:])
	if chunks == []:
		return ""
	return reduce( (lambda x,y : x + " " + y), chunks )

# Se queda con el cuerpo "real" del mail, ignorando los mensajes
# anteriores en el thread.
def get_last_message(body):
	chunks = re.split("-----Original Message-----", body)
	return re.split("####", chunks[0])[0]

# Cuenta links en el mail
def count_links(mail):
	return len(re.findall("(?P<url>https?://[^\s]+)", mail))

# Cuenta cantidad de tags HTML en el mail
def count_html_tags(mail):
	return len(re.findall('<[^<]+?>', mail))

# Cuenta cantidad de caracteres "raros" en el mail
def count_rare_chars(mail): 
    cant = 0
    for x in mail:
        if (1 <= ord(x) and ord(x) <= 32) or (127 <= ord(x) and ord(x) <= 160):
            cant = cant +1
    return cant     

# Elimina caracteres especiales de un string.
# Esto es necesario para agregar los atributos al dataframe:
# cuando contamos las palabras que mas aparecen en un mail,
# las mismas pueden contener caracteres especiales que, al ser
# agregadas al dataframe o posteriormente procesados por algunos
# de los algoritmos, provocaban errores.
def deleteSpecialChars(txt):
    return re.sub('[^A-Za-z0-9]+', '', txt)

def main():
    #################### Inicializacion ####################

    #Esto levanta las listas de los mails desde los archivos
    ham_txt = pickle.load(open("ham_train_dev", "r"))
    spam_txt = pickle.load(open("spam_train_dev", "r"))

    # Intentar levantar el dataframe y la lista de palabras mas utilizadas
    # ya guardados.
    # Si alguno no existe, crearlos.
    try:
        df = pickle.load(open("train/df", "r"))
        try:
            word_count_att_names = pickle.load(open("train/word_count_att_names", "r"))
        except IOError as e:
            raise e
    except IOError:
        # Armo un dataset de Pandas con los mails del dataset 
        # http://pandas.pydata.org/
        df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])

        # Asigno la clase para cada instancia del dataframe
        df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

        # Extraigo atributos simples: 
        # 1) Longitud del mail.
        df['len'] = map(len, df.text)
        # 2) Cantidad de espacios en el mail.
        df['count_spaces'] = map(count_spaces, df.text)
        # 3) Cantidad de links
        df['links'] = map(count_links, df.text)
        # 4) Cantidad de tags
        df['tags'] = map(count_html_tags, df.text)
        # 5) cantidad de caracteres raros
        df['rare'] = map(count_rare_chars, df.text)

        # Conseguimos los cuerpos de cada mail
        all_mail_bodies = ham_txt + spam_txt
        all_mail_bodies[:] = map((lambda x : get_last_message(get_mail_body(x))), all_mail_bodies)

        # Extraemos las palabras mas utilizadas en los cuerpos de los mails y guardamos
        # como atributo la cantidad de apariciones de cada una.
        print "Extrayendo palabras mas usadas"
        count_vectorizer = CountVectorizer(token_pattern='[^\d\W_]\w+', max_features=500)
        word_count_matrix = count_vectorizer.fit_transform(all_mail_bodies)
        word_count_matrix_t = word_count_matrix.transpose().toarray()

        # Guardamos la lista de las 500 palabras mas utilizadas,
        # para luego poder hacer la prediccion sobre los mails.
        word_count_att_names = count_vectorizer.get_feature_names()
        file = open("train/word_count_att_names", "wb")
        pickle.dump(word_count_att_names, file)
        file.close()

        # Preparamos las palabras mas utilizadas para agregarlas al dataframe:
        # eliminamos los caracteres especiales y appendeamos "w_" delante de cada palabra
        # mas usada para evitar posibles conflictos de nombres repetidos (i.e., si una 
        # de las palabras mas usadas fuese "class", ya hay una columna en el dataframe
        # con ese nombre).
        print "Agregando palabras mas usadas como atributos"
        word_count_att_names[:] = map(lambda x : 'w_' + deleteSpecialChars(x), word_count_att_names)
        for att_name_idx in xrange(len(word_count_att_names)):
        	df[word_count_att_names[att_name_idx]] = word_count_matrix_t[att_name_idx]

        # Guardamos el dataframe
        file = open("train/df", "wb")
        pickle.dump(df, file)
        file.close()

    # Preparo data para clasificar
    word_count_att_df_names = map(lambda x : 'w_' + deleteSpecialChars(x), word_count_att_names)
    X = df[['len', 'count_spaces', 'links', 'tags', 'rare'] + word_count_att_df_names].values
    y = df['class']

    #################### Decision Tree Classifier ####################

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
    file = open("resultados/DecisionTreeClf/best_decision_tree_clf", "wb")
    pickle.dump(best_decision_tree_clf, file)
    file.close()
    print np.mean(dt_res), np.std(dt_res)

    #################### Multinomial Naive Bayes Classifier ####################

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
        alpha=mnb_best_params['alpha'],
        fit_prior=mnb_best_params['fit_prior']
        )

    # Ejecutamos el clasificador entrenando con un esquema de CV de 10 folds.
    print "Ejecutando multinomial NB"
    mnb_res = cross_val_score(best_mnb_clf, X, y, cv=10)
    file = open("best_mnb_clf", "wb")
    pickle.dump(best_mnb_clf, file)
    file.close()
    print np.mean(mnb_res), np.std(mnb_res)

    #################### Gaussian Naive Bayes Classifier ####################

    # Creamos un naive bayes gaussiano
    gaussian_nb_clf = GaussianNB()

    # Ejecutamos el clasificador entrenando con un esquema de CV de 10 folds.
    print "Ejecutando gaussian NB"
    gaussian_nb_res = cross_val_score(gaussian_nb_clf, X, y, cv=10)
    print np.mean(gaussian_nb_res), np.std(gaussian_nb_res)

    #################### KNN Classifier ####################

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

    #################### Random Forest Classifier ####################

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

    #################### SVM Classifier ####################

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
    print "Ejecutando SVM"
    svm_res = cross_val_score(best_svm_clf, X, y, cv=10)
    file = open("best_svm_clf", "wb")
    pickle.dump(best_svm_clf, file)
    file.close()
    print np.mean(svm_res), np.std(svm_res)

    #################### Transformacion / Reduccion de dimensionalidad ####################

    print "Aplicando reduccion de dimensionalidad/transformacion de datos"

    # PCA
    print "Aplicando PCA"
    X_PCA = PCA().fit_transform(X, y)

    # Feature selection usando KBest. Usamos chi2 porque estamos usando estructuras
    # esparsas, y la unica funcion que trabaja con ellas sin volverlas densas es chi2.
    # Probamos que cantidad de atributos es mejor, variando el valor entre 20, 50, 100 y 250
    print "Aplicando K-Best selection"
    number_of_attrs = [20, 50, 100, 250]
    classifiers = [best_decision_tree_clf, best_mnb_clf, gaussian_nb_clf, best_knn_clf, best_rf_clf, best_svm_clf]
    for n_attrs in number_of_attrs:
        for clf in classifiers:
            X_new = SelectKBest(chi2, k=n_attrs).fit_transform(X_PCA, y)
            print "Ejecutando " + type(clf).__name__ + " con mejores " + str(n_attrs) + " atributos..."
            best_attrs_res = cross_val_score(clf, X_new, y, cv=10)
            print np.mean(best_attrs_res), np.std(best_attrs_res)

if __name__ == "__main__":
    main()