import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from itertools import cycle


def plot_multiclass_roc_auc(y, n_classes: int, y_prob: np.ndarray):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y.ravel(), y_prob.ravel())
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcula el micro-average AUC (promedio de todos los puntos)
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Grafica las curvas ROC para cada clase
    plt.figure()
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=f'ROC curve (area = {roc_auc[i]:.2f})')

    # Grafica la curva ROC micro-average
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=lw,
            label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multiclass Classification')
    plt.legend(loc="lower right")
    return plt.gcf()
