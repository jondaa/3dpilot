import torch
from sklearn.cluster import KMeans
import numpy as np


def find_closest_point(center, points):
    aftersub=points-center
    return torch.argmin(torch.sum(aftersub**2,dim=1)).int()


def balanced_kmeans(num_clusters, points):
    kmeans = KMeans(n_clusters=num_clusters).fit(points)
    clusters = torch.zeros(num_clusters,points.shape[0]//num_clusters,3)
    clusters_centers = torch.from_numpy(kmeans.cluster_centers_).float()
    for i in range(clusters.shape[1]):
        for cluster_idx, cluster_value in enumerate(clusters_centers):
            closest_point_idx = find_closest_point(cluster_value, points)
            clusters[cluster_idx,i,:]=(points[closest_point_idx,:])
            points = torch.cat([points[0:closest_point_idx], points[closest_point_idx+1:]])
            clusters_centers[cluster_idx,:] = clusters[cluster_idx,:,:].mean(dim=0)
    return clusters

#
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     num_shots = 8
#     num_measurements_shot = 100
#     x = torch.randn(num_measurements_shot*num_shots, 3)
#     clusters = balanced_kmeans(num_clusters=num_shots, points=x)
#     fig = plt.figure(figsize=[10, 10])
#     ax = plt.axes(projection='3d')
#     colors = ['g.', 'r.', 'b.', 'y.', 'c.','m.','k.']
#     clusters=clusters.numpy()
#     for klass, color in zip(range(0, clusters.shape[0]), colors):
#         ax.plot3D(clusters[klass, :, 0], clusters[klass, :, 1], clusters[klass, :, 2],color)
#
#     ax.set_title('Kmeans Clustering\n')
#     plt.show()