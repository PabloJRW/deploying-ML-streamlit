import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_multiclass_roc_auc(data, y_test, y_prob):
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