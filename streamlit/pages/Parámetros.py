import os
import sys
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Obtener la ruta del directorio "Iris" (directorio padre)
ruta_padre = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Agregar la ruta del directorio "Iris" al PYTHONPATH
sys.path.append(ruta_padre)

from functions.classifier import get_clf
from functions.plot import plot_multiclass_roc_auc
from functions.scores import get_scores
from functions.params_selector import add_params


st.title('Predicción de especies de Iris')


st.sidebar.header("Parámetros")

clf_selected = st.sidebar.selectbox("Selecciona un algoritmo:", ("Random Forest", "KNN", "SVM"))


params = add_params(clf_selected)

      

clf = get_clf(clf_selected, params)

data = pd.read_csv(os.path.join('datasets','raw','Iris.csv'))
data.drop('Id', axis=1, inplace=True)

X = data.drop('Species', axis=1)
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)



y_prob = clf.predict_proba(X_test)

#roc_auc_plot = plot_multiclass_roc_auc(data, y_test, y_prob)

scores = get_scores(y_test, y_pred)



st.subheader("Puntajes de predicción")
st.write(f"{clf_selected}: ")
st.dataframe(data=scores)
st.write(scores)  