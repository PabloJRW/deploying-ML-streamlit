import streamlit as st


st.title('Predicción de especies de Iris')

col1, col2 = st.columns(2)

with col1:
    st.subheader("Introduzca los valores")
    sepal_length = st.number_input('Longitud del sépalo')
    sepal_width = st.number_input('Anchura del sépalo')
    petal_length = st.number_input('Longitud del pétalo')
    petal_width = st.number_input('Anchura del pétalo')


with col2:
    st.subheader("Predicción")
    st.write(f"Clase:")
    