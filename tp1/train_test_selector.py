import json
import pickle
import random

dataset_path = ''

ham_txt = json.load(open(dataset_path + 'ham_dev.json'))
spam_txt = json.load(open(dataset_path + 'spam_dev.json'))

test_txt = random.sample(ham_txt + spam_txt, 20000)

#saco de ham_txt y spam_txt los mails que seleccione para test
ham_txt = [m for m in ham_txt if m not in test_txt]
spam_txt = [m for m in spam_txt if m not in test_txt]

#guardo las listas en archivos para despues levantarlas directamente, no trabajamos mas en formato JSON despues de correr este script
file = open("ham_train_dev", "wb")
pickle.dump(ham_txt, file)
file.close()

file = open("spam_train_dev", "wb")
pickle.dump(spam_txt, file)
file.close()

file = open("test_dev", "wb")
pickle.dump(test_txt, file)
file.close()

#ham_txt = pickle.load(open("ham_train_dev", "rb"))
#spam_txt = pickle.load(open("spam_train_dev", "rb"))
#test_txt = pickle.load(open("test_dev", "rb"))