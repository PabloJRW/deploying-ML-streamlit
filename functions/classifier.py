from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_clf(clf_selected, params):
    # Verifica qué algoritmo de clasificación fue seleccionado
    if clf_selected == "Random Forest":
        # Si es "Random Forest", crea un clasificador RandomForestClassifier
        # utilizando los parámetros n_estimators, max_depth y random_state
        clf = OneVsRestClassifier(RandomForestClassifier(n_estimators = params["n_estimators"],
                                     max_depth = params["max_depth"],
                                     random_state=42))
    elif clf_selected == "KNN":
        # Si es "KNN", crea un clasificador KNeighborsClassifier
        # utilizando el parámetro n_neighbors
        clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = params["K"]))
    elif clf_selected == "SVM":
        # Si es "SVM", crea un clasificador SVC (Support Vector Classifier)
        # utilizando los parámetros kernel y C
        clf = OneVsRestClassifier(SVC(kernel=params["kernel"],
                  C=params["C"]))
    else: 
        # Si el algoritmo seleccionado no es válido, genera un error
        raise ValueError("Algoritmo no válido")
    
    # Devuelve el clasificador creado
    return clf   