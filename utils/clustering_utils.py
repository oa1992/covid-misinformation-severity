import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

#pca = PCA(n_components = 2) # MAYBE NO?
def create_dendrogram(dataset: "The topic embeddings."):
    dendrogram = sch.dendrogram(sch.linkage(dataset, method='ward'))

    model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    model.fit(dataset)

    labels = model.labels_

    return labels
