import numpy as np
import streamlit as st

import joblib


# Importación del modelo random forest
rf_model = joblib.load("rf_model.joblib")


###################################################
st.title('Predicción de especies de Iris')
  

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
rf_pred = rf_model.predict(inputs_to_pred)

# Resultados de predicción
with col3:
    st.subheader("Predicción")
    st.markdown(f"Random Forest Predicted Class: **{rf_pred[0]}**")
    st.write(f"Logistic Regression Predicted Class: ")
    st.write(f"KNN Predicted Class: ")
    st.write(f"SVM Forest Predicted Class: ")
    
