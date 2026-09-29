import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay, f1_score,\
    roc_auc_score, accuracy_score, confusion_matrix, RocCurveDisplay


# Functions
def pred_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rocauc = roc_auc_score(y_test, y_pred)
    print(f'acc: {round(acc, 4)}, f1: {round(f1,4)}, roc_auc: {round(rocauc,4)}')
    print(confusion_matrix(y_test, y_pred))
    return



