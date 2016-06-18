from api import app
from flask import jsonify
from flask import render_template
from sklearn import preprocessing
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


@app.route('/head')
def head():
    df = get_data().head()
    data = json.loads(df.to_json())
    return jsonify(data)


@app.route('/count')
def count():
    pass
