import streamlit as st

def add_params(clf_selected):
    params = dict()
    if clf_selected == "Random Forest":
        with st.sidebar.expander("Random Forest"):
            n_estimators = st.slider("Elige la cantidad de estimadores:", min_value=2, max_value=12, key="rf_estimatores")
            max_depth = st.slider("Elige la profundidad máxima:", min_value=2, max_value=12, key="rf_maxDepth")
            params["n_estimators"] = n_estimators
            params["max_depth"] = max_depth
    elif clf_selected == "KNN":
        with st.sidebar.expander("KNN"):
            K = st.slider("K", min_value=1, max_value=10, key="knn_k")
            params["K"] = K
    elif clf_selected == "SVM":
        with st.sidebar.expander("SVM"):
            kernel = st.selectbox("Elige el kernel:", ("linear", "poly", "RBF", "sigmoid"), key="knn_kernel")
            C = st.slider("C", min_value=0.01, max_value=10.0, key="knn_c")
            params["kernel"] = kernel
            params["C"] = C

    return params

