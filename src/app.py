from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import classification_report
from pickle import dump

#CARGAR DATOS
url= "https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv"
datos= pd.read_csv(url)
datos.head()
print(datos.columns)
datos

#ARREGLAR DATOS SEGUN EJERCICIO
datos= datos.drop(["package_name"], axis= 1)

datos["review"]= datos["review"].str.strip().str.lower()
datos["review"] = datos["review"].str.replace(r'\s+', ' ', regex=True)

#TRAIN Y TEST
X= datos["review"]
y=datos["polarity"]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

X_train.head()

X_test.head()

#VECTORIZACIÓN Y MATRIZ DE CONTEO 
vectorizer = CountVectorizer(stop_words = "english")
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()
X_train_vec

X_test_vec

palabras= vectorizer.get_feature_names_out()
print(palabras)
#NO ENTIENDO PORQUE ME SALEN ESAS PALABRAS SI EN EL CUADRO DE TEXTO 49 Y 50 ME APARECEN PALABRAS NORMALES 

#IMPLANTAR MODELOS DE NAIVEBAYES
#Multinomial
mnb=MultinomialNB()
mnb.fit(X_train_vec, y_train)
y_pred_m= mnb.predict(X_test_vec)
y_pred_m

print(f"La métrica del modelo Multinomial es:")
print(classification_report(y_test, y_pred_m))

#IMPLANTAR MODELOS DE NAIVEBAYES
#GaussianNB
gnb=GaussianNB()
gnb.fit(X_train_vec, y_train)
y_pred_g= gnb.predict(X_test_vec)
y_pred_g

print(f"La métrica del modelo Gaussiano es:")
print(classification_report(y_test, y_pred_g))

#IMPLANTAR MODELOS DE NAIVEBAYES
#BernuolliNB
bnb=BernoulliNB()
bnb.fit(X_train_vec, y_train)
y_pred_b= bnb.predict(X_test_vec)
y_pred_b


print(f"La métrica del modelo Bernuolli es:")
print(classification_report(y_test, y_pred_b))

Tras implantar los 3 modelos de NaiveBayes. El modelo Multinomial es el que tiene una mejor exactitud con un valor = 0.82

#OPTIMIZAR MULTINOMINALNB:
hiperparametros = {"alpha":[0.001, 0.01, 0.1, 1, 10, 100], "fit_prior": [True, False]}

opti = RandomizedSearchCV(mnb, hiperparametros, random_state = 42)
opti.fit(X_train_vec, y_train)

print(opti.best_params_)

mnb_opti = MultinomialNB(alpha= 1, fit_prior=False)
mnb_opti.fit(X_train_vec, y_train)
y_pred_o= mnb_opti.predict(X_test_vec)
print(f"La métrica del modelo Multinomial Optimizado es:")
print(classification_report(y_test, y_pred_o))


La accuracy ha bajado un 0.02 en el modelo optimizado, no se si esto puede significar que es mejor una accuracy de 0.80 vs 0.82 para evitar el overfitting

#Guardar modelo 
dump(mnb_opti, open("/workspace/NaiveBayes/models/modelo_optimizado_multinomialNB.sav", "wb"))
