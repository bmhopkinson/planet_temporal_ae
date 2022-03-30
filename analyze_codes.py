import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans, DBSCAN, OPTICS, SpectralClustering
from sklearn.manifold import TSNE
import umap

import matplotlib.pyplot as plt
import matplotlib

infile = 'encoded_30m_data.txt'
cluster_alg_choice = 'KMeans'  #DBSCAN, KMeans, OPTICS, Spectral
embed_choice = 'UMAP'
cmap = plt.get_cmap('turbo')  #'gist_ncar' is a little wild but good for distinguishing lots of clusters,

data = np.loadtxt(infile)
code = data[:,2:]

# Dimensionality reduction via PCA
scaler = StandardScaler()
pca = PCA(n_components=4)
X = scaler.fit_transform(code)
pca.fit(X)
print('hello')

features = pca.transform(X)

# Clustering
if cluster_alg_choice == 'KMeans':
    cluster_alg = KMeans(n_clusters=10, random_state=0)
    cluster_input = features
elif cluster_alg_choice == 'DBSCAN':
    cluster_alg = DBSCAN(eps=1.0, min_samples=10)
    cluster_input = features
elif cluster_alg_choice == 'OPTICS':
    cluster_alg = OPTICS(min_samples=10)
    cluster_input = features
elif cluster_alg_choice == 'Spectral':
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(features, metric='euclidean')
    sigma = 0.5
    A = np.exp((-1/(2*sigma**2)) * D)  #gaussian
    cluster_input = A
    cluster_alg = SpectralClustering(n_clusters=10, affinity='precomputed')

else:
    print('error: cluster algorithm not recognized')
cluster_results = cluster_alg.fit(cluster_input)
cluster_ids = cluster_results.labels_
unique_cids =np.unique(cluster_ids)
ncids = len(unique_cids)

#define colors for plotting
colors = [cmap((i/ncids)) for i in range(0, cluster_ids.max()+1)]
intervals = np.arange(-0.5, cluster_ids.max()+1.5, 1.0)
cmap, norm = matplotlib.colors.from_levels_and_colors(intervals, colors)

#plot clusters in physical space
fig, ax = plt.subplots(1, 1)
scatter_data = ax.scatter(data[:, 1], data[:, 0], s=0.2, c=cluster_ids, cmap=cmap,
                          norm=norm, marker='o', linewidth=0, alpha=1.0)
ax.legend(*scatter_data.legend_elements(), fontsize='xx-small')
ax.set_aspect('equal', 'box')

plt.savefig('satellite_ts_clusters_physical_space.png', dpi=600)

# embed in low dimension for visualization
if embed_choice == 'tSNE':
    tsne = TSNE(n_components=2, learning_rate='auto', init='random')
    X_embedded = tsne.fit_transform(features)
elif embed_choice == 'UMAP':
    umap = umap.UMAP()
    X_embedded = umap.fit_transform(features)
else:
    print('no valid embedding, skipping')

#plot clusters in embedded space
if X_embedded is not None:
    fig = plt.figure()
    ax = plt.gca()
    scatter_data2 = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=cluster_ids, cmap=cmap, norm=norm, marker='.', alpha=0.5)
    ax.legend(*scatter_data2.legend_elements(), fontsize='xx-small')
    plt.savefig('satellite_ts_embedded_space.jpg', dpi=600)
