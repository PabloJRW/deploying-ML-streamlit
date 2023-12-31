import numpy as np
import streamlit as st

import pickle


# Importación de modelos
with open('trained_models/rf_model.pkl', 'rb') as model: # Random Forest
    rf_clf = pickle.load(model) 

with open('trained_models/lr_model.pkl', 'rb') as model: # Logistic Regression
    lr_clf = pickle.load(model)  

with open('trained_models/knn_model.pkl', 'rb') as model: # KNN
    knn_clf = pickle.load(model) 

with open('trained_models/svm_model.pkl', 'rb') as model: # SVM
    svm_clf = pickle.load(model) 


################################################################################
st.title('Predicción de especies de Iris')

st.info("Puedes predecir la clase de Iris ingresando los varoles de las variables (Ancho de Sépalo, \
        Largo de Sépalo, Ancho de Pétalo, Largo de Pétalo)")
  

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Introduzca los valores")
    sepal_length = st.slider('Longitud del sépalo', min_value=4.3, max_value=7.9, value=5.84)
    sepal_width = st.slider('Anchura del sépalo', min_value=2.0, max_value=4.4, value=3.05)
    petal_length = st.slider('Longitud del pétalo', min_value=1.0, max_value=6.9, value=3.75)
    petal_width = st.slider('Anchura del pétalo', min_value=0.1, max_value=2.5, value=1.19)


# Arreglo de los valores introducidos
inputs_to_pred = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

# Predicción
rf_pred = rf_clf.predict(inputs_to_pred)
lr_pred = lr_clf.predict(inputs_to_pred)
knn_pred = knn_clf.predict(inputs_to_pred)
svm_pred = svm_clf.predict(inputs_to_pred)


# Resultados de predicción
boton = st.button("Predecir")

if boton:
    with col3:
        st.subheader("Predicción")
        st.markdown(f"Random Forest: ")
        st.success(rf_pred[0])
        st.markdown(f"Logistic Regression: ")
        st.success(lr_pred[0])
        st.write(f"KNN: ")
        st.success(knn_pred[0])
        st.write(f"SVM: ")
        st.success(svm_pred[0])
    
