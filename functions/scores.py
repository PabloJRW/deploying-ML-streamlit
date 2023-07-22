from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def get_scores(y_test, y_pred):
    #acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='micro')
    rec = recall_score(y_test, y_pred, average='micro')
    
    scores = {'Precision':prec, 'Recall':rec}
    return scores