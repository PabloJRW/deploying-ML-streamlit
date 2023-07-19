import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


st.title('Predicción de especies de Iris')


st.sidebar.header("Parámetros")

clf_selected = st.sidebar.selectbox("Selecciona un algoritmo:", ("Random Forest", "KNN", "SVM"))

def add_params(clf_selected):
    params =dict()
    if clf_selected == "Random Forest":
        n_estimators = st.sidebar.slider("Elige la cantidad de estimadores:", min_value=2, max_value=12)
        max_depth = st.sidebar.slider("Elige la profundidad máxima:", min_value=2, max_value=12)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
    elif clf_selected == "KNN":
        K = st.sidebar.slider("K", min_value=1, max_value=10)
        params["K"] = K
    elif clf_selected == "SVM":
        kernel = st.sidebar.selectbox("Elige el kernel:", ("linear", "poly", "RBF", "sigmoid"))
        C = st.sidebar.slider("C", min_value=0.01, max_value=10.0)
        params["kernel"] = kernel
        params["C"] = C

    return params

params = add_params(clf_selected)
    
def get_clf(clf_selected, params):
    if clf_selected == "Random Forest":
        clf = OneVsRestClassifier(RandomForestClassifier(n_estimators = params["n_estimators"],
                                     max_depth = params["max_depth"],
                                     random_state=42))
    elif clf_selected == "KNN":
        clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = params["K"]))
    elif clf_selected == "SVM":
        clf = OneVsRestClassifier(SVC(kernel=params["kernel"],
                  C=params["C"]))
    else: 
        raise ValueError("Algoritmo no válido")
    
    return clf    

clf = get_clf(clf_selected, params)

data = pd.read_csv(os.path.join('datasets','raw','Iris.csv'))
data.drop('Id', axis=1, inplace=True)

X = data.drop('Species', axis=1)
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = round(precision_score(y_test, y_pred, average='micro'), 3)
recall = round(recall_score(y_test, y_pred, average='micro'), 3)
f1 = round(f1_score(y_test, y_pred, average='micro'), 3)

y_prob = clf.predict_proba(X_test)

# Calcular la curva ROC y el AUC para cada clase
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(data['Species'].unique())):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcular el promedio del AUC para todas las clases
mean_auc = np.mean(list(roc_auc.values()))

# Graficar la curva ROC para cada clase
plt.figure()
for i in range(len(data["Species"].unique())):
    plt.plot(fpr[i], tpr[i], lw=2, label='Curva ROC para clase {} (AUC = {:.2f})'.format(data['Species'][i], roc_auc[i]))

# Línea base (diagonal)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC para Clasificación Multiclase (Promedio AUC = {:.2f})'.format(mean_auc))
plt.legend(loc="lower right")


col1, col2 = st.columns(2)

with col1:
    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)


with col2:
    st.subheader("Puntajes de predicción")
    st.write(f"{clf_selected}: ")
    st.dataframe(data={'Accuracy' :accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1})
