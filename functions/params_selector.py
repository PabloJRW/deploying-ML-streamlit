import streamlit as st

def add_params(clf_selected):
    params = dict()
    if clf_selected == "Random Forest":
        n_estimators = st.sidebar.slider("Elige la cantidad de estimadores:", min_value=2, max_value=12, key="rf_estimatores")
        max_depth = st.sidebar.slider("Elige la profundidad máxima:", min_value=2, max_value=12, key="rf_maxDepth")
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
    elif clf_selected == "KNN":
        K = st.sidebar.slider("K", min_value=1, max_value=10, key="knn_k")
        params["K"] = K
    elif clf_selected == "SVM":
        kernel = st.sidebar.selectbox("Elige el kernel:", ("linear", "poly", "RBF", "sigmoid"), key="knn_kernel")
        C = st.sidebar.slider("C", min_value=0.01, max_value=10.0, key="knn_c")
        params["kernel"] = kernel
        params["C"] = C

    return params

   