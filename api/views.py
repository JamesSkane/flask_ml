from api import app
from flask import jsonify
from flask import render_template
from flask import url_for
import json
import os
import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline


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


@app.route('/')
def index():
    df = get_data()
    X = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()
    # Scale
    scaler = preprocessing.StandardScaler().fit(X)
    scaled = scaler.transform(X)
    # PCA
    pcomp = decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components = pcomp.transform(scaled)
    var = pcomp.explained_variance_ratio_.sum() # View explained var w/ debug
    # Kmeans
    model = KMeans(init='k-means++', n_clusters=2)
    model.fit(components)
    # Plot
    fig = plt.figure()
    plt.scatter(components[:, 0], components[:, 1], c=model.labels_)
    centers = plt.plot(
        [model.cluster_centers_[0, 0], model.cluster_centers_[1, 0]],
        [model.cluster_centers_[1, 0], model.cluster_centers_[1, 1]],
        'kx', c='Green'
    )
    # Increase size of center labels
    plt.setp(centers, ms=11.0)
    plt.setp(centers, mew=1.8)
    axes = plt.gca()
    axes.set_xlim([-7.5, 3])
    axes.set_ylim([-2, 5])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clustering of PCs ({:.2f}% Var. Explained)'.format(var * 100))
    fig_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cluster.png')
    fig.savefig(fig_path)
    return render_template('index.html',
                           fig=url_for('static', filename='tmp/cluster.png'))


@app.route('/d3')
def d3():
    df = get_data()
    X = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()
    # Scale
    scaler = preprocessing.StandardScaler().fit(X)
    scaled = scaler.transform(X)
    # PCA
    pcomp = decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components = pcomp.transform(scaled)
    var = pcomp.explained_variance_ratio_.sum() # View explained var w/ debug
    # Kmeans
    model = KMeans(init='k-means++', n_clusters=2)
    model.fit(components)
    # Generate CSV
    cluster_data = pd.DataFrame({'pc1': components[:, 0],
                                 'pc2': components[:, 1],
                                 'labels': model.labels_})
    csv_path = os.path.join(get_abs_path(), 'static', 'tmp', 'kmeans.csv')
    cluster_data.to_csv(csv_path)
    return render_template('d3.html',
                           data_file=url_for('static',
                                             filename='tmp/kmeans.csv'))

@app.route('/api/v1/prediction_confusion_matrix')
def confusion_mat():
    frame = get_data()
    X = frame.ix[:, (frame.columns != 'class') & (frame.columns != 'code')].as_matrix()
    y = frame.ix[:, frame.columns == 'class'].as_matrix()
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y.ravel())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    pipe_lr = Pipeline([('scl', preprocessing.StandardScaler()),
                        ('pca', decomposition.PCA(n_components=2)),
                        ('clf', LogisticRegression(random_state=1))])
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    #data = json.load(confmat.to_json())
    #return jsonify(confmat)
    c = pd.DataFrame([val for row in confmat for val in row], index=['tp', 'fn', 'fp', 'tn'],columns=['logistic regression'])
    data = json.loads(c.to_json())
    return jsonify(data)


@app.route('/prediction')
def classify():
    frame = get_data()
    X = frame.ix[:, (frame.columns != 'class') & (frame.columns != 'code')].as_matrix()
    y = frame.ix[:, frame.columns == 'class'].as_matrix()
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y.ravel())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    pipe_lr = Pipeline([('scl', preprocessing.StandardScaler()),
                        ('pca', decomposition.PCA(n_components=2)),
                        ('clf', LogisticRegression(random_state=1))])
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    cv = StratifiedKFold(y, n_folds=6)
    classifier = pipe_lr
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random chance')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic with cross validation')
    plt.legend(loc="lower right")
    fig_path = os.path.join(get_abs_path(), 'static', 'tmp', 'roc.png')
    fig.savefig(fig_path)
    return render_template('prediction.html',
                           fig=url_for('static', filename='tmp/roc.png'))


@app.route('/head')
def head():
    df = get_data().head()
    data = json.loads(df.to_json())
    return jsonify(data)


@app.route('/count')
def count():
    pass


@app.route('/cm')
def cm():
    frame = get_data()
    X = frame.ix[:, (frame.columns != 'class') & (frame.columns != 'code')].as_matrix()
    y = frame.ix[:, frame.columns == 'class'].as_matrix()
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y.ravel())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    pipe_lr = Pipeline([('scl', preprocessing.StandardScaler()),
                        ('pca', decomposition.PCA(n_components=2)),
                        ('clf', LogisticRegression(random_state=1))])
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i,
                    s=confmat[i, j],
                    va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('Confusion Matrix: Logistic Regression Pipeline')
    fig_path = os.path.join(get_abs_path(), 'static', 'tmp', 'con_mat.png')
    fig.savefig(fig_path)
    return render_template('index.html', fig=url_for('static', filename='tmp/con_mat.png'))