from api import app
from flask import jsonify
from flask import render_template
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
from scipy import interp
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.cross_validation import StratifiedKFold

from sklearn import decomposition
from sklearn.cluster import KMeans
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import url_for

def get_abs_path():
    """
    Get the absolute path of the Flask project.

    Returns
    -------
    str
        The absolute path of the project in the format appropriate for the
        operating system.
    """
    return os.path.abspath(os.path.dirname(__file__))


def get_data():
    f_name = os.path.join(get_abs_path(), 'data', 'breast-cancer-wisconsin.csv')
    columns = ['code', 'clump_thickness', 'size_uniformity', 'shape_uniformity',
               'adhesion', 'cell_size', 'bare_nuclei', 'bland_chromatin',
               'normal_nuclei', 'mitosis', 'class']
    df = pd.read_csv(f_name, sep=',', header=None, names=columns, na_values='?')
    return df.dropna()


def prep_data(frame):
    X = frame.ix[:, (frame.columns != 'class') & (frame.columns != 'code')].as_matrix()
    y = frame.ix[:, frame.columns == 'class'].as_matrix()
    le = LabelEncoder()
    y = le.fit_transform(y.ravel())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    pipe_lr = Pipeline([('scl', StandardScaler()),
                        ('pca', PCA(n_components=2)),
                        ('clf', LogisticRegression(random_state=1))])
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i,
                    s=confmat[i, j],
                    va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show(1)
    cv = StratifiedKFold(y, n_folds=6)
    classifier = pipe_lr

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show(1)



df = get_data()
prep_data(df)
