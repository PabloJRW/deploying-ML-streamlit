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

# DATA
###############################################################################
data = pd.read_csv(os.path.join('datasets','raw','Iris.csv'))
data.drop('Id', axis=1, inplace=True)

X = data.drop('Species', axis=1)
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, random_state=42)


################################################################################
st.title('Predicción de especies de Iris')


# SIDEBAR
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


#  
clf = get_clf(clf_selected, params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)



#y_prob = clf.predict_proba(X_test)

#roc_auc_plot = plot_multiclass_roc_auc(data, y_test, y_prob)

scores = get_scores(y_test, y_pred)



st.subheader("Puntajes de predicción")
st.write(f"{clf_selected}: ")
st.table(data=scores)
