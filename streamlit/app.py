import streamlit as st


st.title('Predicción de especies de Iris')
sepal_length = st.number_input('Longitud del sépalo')
sepal_width = st.number_input('Anchura del sépalo')
petal_length = st.number_input('Longitud del pétalo')
petal_width = st.number_input('Anchura del pétalo')

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
##prediction = model.predict(input_data)

#species = iris.target_names[prediction[0]]
#st.write(f'La especie de Iris predicha es: {species}')  




st.number_input('dji')