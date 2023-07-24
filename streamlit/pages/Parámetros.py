import os
import sys

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Obtener la ruta del directorio "Iris" (directorio padre)
ruta_padre = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Agregar la ruta del directorio "Iris" al PYTHONPATH
sys.path.append(ruta_padre)

from functions.classifier import get_clf
from functions.scores import get_scores


# DATA
###############################################################################
ruta = os.path.join("datasets","raw")
data = pd.read_csv(os.path.join(ruta,'Iris.csv'))
data.drop('Id', axis=1, inplace=True)

X = data.drop('Species', axis=1)
y = data['Species']
y_labels = y.unique()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, random_state=42)

# STREAMLIT APP
################################################################################
st.title('Predicción de especies de Iris')
st.info("En esta sección puedes elegir un modelo (Random Forest, Logistic Regression, KNN, SVM),\
            modificar sus parámetros y ver en las métricas cómo se ajusta el modelo.")

 
# STREAMLIT SIDEBAR
#################################################################################
st.sidebar.header("Parámetros")
clf_selected = st.sidebar.selectbox("Selecciona un algoritmo:", ("Random Forest", "Logistic Regression", "KNN", "SVM"))
params = dict()
if clf_selected == "Random Forest":
    n_estimators = st.sidebar.slider("Elige la cantidad de estimadores:", min_value=2, max_value=12, key="rf_estimatores")
    max_depth = st.sidebar.slider("Elige la profundidad máxima:", min_value=2, max_value=12, key="rf_maxDepth")
    params["n_estimators"] = n_estimators
    params["max_depth"] = max_depth
elif clf_selected == "Logistic Regression":
    solver = st.sidebar.selectbox("Selecciona el solver:", ("lbfgs","newton-cg","sag","saga"))
    max_iter = st.sidebar.slider("Asigna el máximo de iteraciones:", min_value=5, max_value=15)
    C = st.sidebar.slider("Asigna el valor de C:", min_value=0.01, max_value=1.0)
    params["solver"] = solver
    params["max_iter"] = max_iter
    params['C'] = C
elif clf_selected == "KNN":
    K = st.sidebar.slider("K", min_value=1, max_value=10, key="knn_k")
    params["K"] = K
elif clf_selected == "SVM":
    kernel = st.sidebar.selectbox("Elige el kernel:", ("linear", "poly", "RBF", "sigmoid"), key="knn_kernel")
    C = st.sidebar.slider("C", min_value=0.01, max_value=10.0, key="knn_c")
    params["kernel"] = kernel
    params["C"] = C


# ############################################################################


# MODELO
clf = get_clf(clf_selected, params) # Instancia del modelo y sus parámetros
clf.fit(X_train, y_train) # Entrenando el modelo...
y_pred = clf.predict(X_test) # Predicción

# Puntajes del modelo
scores = get_scores(y_test, y_pred)


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)  
fig = plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y_labels, yticklabels=y_labels)
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')
plt.title('Matriz de Confusión')
plt.tight_layout()


# ELEMENTOS STREAMLIT
col1, col2 = st.columns(2)
with col1:
    st.subheader("Matriz de confusión")
    st.write(fig)

with col2:
    st.subheader("Puntajes de predicción")
    st.dataframe(scores)

