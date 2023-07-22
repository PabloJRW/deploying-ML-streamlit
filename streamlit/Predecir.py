import streamlit as st
import os
import joblib
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib


# DATA
###############################################################################

data = pd.read_csv(os.path.join('datasets','raw','Iris.csv'))
data.drop('Id', axis=1, inplace=True)

X = data.drop('Species', axis=1)
y = data['Species']

# Binariza las etiquetas utilizando One-vs-Rest
y_bin = label_binarize(y, classes=['Iris-setosa', 'Iris-versicolor', 'Iris-Virginica'])
n_classes = y_bin.shape[1]


X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)


###################################################


st.title('Predicción de especies de Iris')
  








col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Introduzca los valores")
    sepal_length = st.slider('Longitud del sépalo', min_value=4.3, max_value=7.9, value=5.84)
    sepal_width = st.slider('Anchura del sépalo', min_value=2.0, max_value=4.4, value=3.05)
    petal_length = st.slider('Longitud del pétalo', min_value=1.0, max_value=6.9, value=3.75)
    petal_width = st.slider('Anchura del pétalo', min_value=0.1, max_value=2.5, value=1.19)


import numpy as np
inputs_to_pred = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
rf_pred = rf_clf.predict(inputs_to_pred)
st.write(inputs_to_pred)
st.write(rf_pred)

with col3:
    st.subheader("Predicción")
    st.write(f"Random Forest Predicted Class:")
    st.write(f"Logistic Regression Predicted Class: ")
    st.write(f"KNN Predicted Class: ")
    st.write(f"SVM Forest Predicted Class: ")
    
