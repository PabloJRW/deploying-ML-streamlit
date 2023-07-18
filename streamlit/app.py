import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


st.title('Predicción de especies de Iris')
sepal_length = st.number_input('Longitud del sépalo')
sepal_width = st.number_input('Anchura del sépalo')
petal_length = st.number_input('Longitud del pétalo')
petal_width = st.number_input('Anchura del pétalo')

#input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

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
        clf = RandomForestClassifier(n_estimators = params["n_estimators"],
                                     max_depth = params["max_depth"],
                                     random_state=42)
    elif clf_selected == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params["K"])
    elif clf_selected == "SVM":
        clf = SVC(kernel=params["kernel"],
                  C=params["C"])
    else: 
        raise ValueError("Algoritmo no válido")
    
    return clf    

clf = get_clf(clf_selected, params)

 