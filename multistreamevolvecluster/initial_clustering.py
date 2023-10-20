import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial as spt
import pickle
import os

from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import zscore
from functions import kmedoids, IC_av, calculate_connectivity, pairwise_euclidean
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances

def initial_kmedoids_clustering(data, mini, maxi, iterations, distances=None, centroids=None):
	if distances is None:
		distances = calculate_distances(data.to_numpy(copy=True))

	silhouette_scores, silhouette_clusters = [],[]
	ic_scores, ic_clusters = [],[]
	connectivity_scores, connectivity_clusters = [],[]
	for num_clusters in range(mini, maxi):
		best_silhouette, best_ic, best_connectivity = None, None, None
		sil_cluster, ic_cluster, conn_cluster = None, None, None
		
		print('num clusters: ',num_clusters)
		j = 0
		while j < iterations:
			j += 1
			try:
				C, M = kmedoids(distances, num_clusters)
			except Exception as e:
				continue
			clusters = [None for x in range(len(data))]
			for i in range(len(M)):
				for index in C[i]:
					clusters[index] = M[i]
			try:
				silhouette = silhouette_score(distances, clusters, metric='precomputed')
				if best_silhouette is None or silhouette > best_silhouette:
					best_silhouette = silhouette
					sil_cluster = clusters

				connectivity = calculate_connectivity(data, 
													clusters,
													[x for x in range(data.shape[1])],
													10, distance_matrix=distances)['CONN'].sum()
				if best_connectivity is None or connectivity < best_connectivity:
					best_connectivity = connectivity
					conn_cluster = clusters

				ic = IC_av(distances, clusters)[0]
				if best_ic is None or ic < best_ic:
					best_ic = ic
					ic_clusters = clusters
			except:
				continue

		silhouette_scores.append(best_silhouette)
		silhouette_clusters.append(sil_cluster)
		ic_scores.append(best_ic)
		ic_clusters.append(ic_cluster)
		connectivity_scores.append(best_connectivity)
		connectivity_clusters.append(conn_cluster)

	return [silhouette_scores,
			silhouette_clusters,
			ic_scores,
			ic_clusters,
			connectivity_scores,
			connectivity_clusters]

def ampds_multiple_distances():
	files = ['elec', 'water', 'gas', 'weather', 'all']
	distance_functions = ['canberra', 'chebyshev', 'correlation', 'cosine', 'mahalanobis', 'braycurtis', 'minkowski']
	mini = 2
	maxi = 11
	iterations = 250
	hours = [1,2,3,4,6,8]

	for hour in hours:
		for k in range(1):
		#for k in range(12):
			for i in range(len(distance_functions)):
				for file in files:
					print("Starting", file)
					data = pd.read_csv(f'data/AMPds2/{hour}H_{file}_segment_{k}.csv', index_col=0, parse_dates=True)
					#data = data.apply(zscore, axis=1, result_type='expand')
					distances = pd.DataFrame(pairwise_distances(data.to_numpy(copy=True), metric=distance_functions[i]))

					returns = initial_kmedoids_clustering(data.to_numpy(copy=True), mini, maxi, iterations, distances.to_numpy(copy=True))

					index = [x for x in range(mini, maxi)]
					cols = ["sil", "sil_clusters", "ic", "ic_clusters", "conn", "conn_clusters"]
					df = pd.DataFrame(columns = cols)
					for j in range(len(returns)):
						df[cols[j]] = returns[j]
					df.index = index
					df.to_csv(f'data/AMPds2/initial_clusterings/{hour}H_initial_{file}_{distance_functions[i]}_seg_{k}.csv')

def ampds():
	files = ['elec', 'water', 'gas', 'weather', 'all']
	distance_functions = [pairwise_euclidean, fastdtw_wrapper]
	mini = 2
	maxi = 11
	iterations = 250
	hours = [1,2,3,4,6,8]
	hours = [8]

	for hour in hours:
		for k in range(1):
		#for k in range(12):
			for i in range(len(distance_functions)):
				for file in files:
					print("Starting", file)
					data = pd.read_csv(f'data/AMPds2/{hour}H_{file}_segment_{k}.csv', index_col=0, parse_dates=True)
					#data = data.apply(zscore, axis=1, result_type='expand')
					distances = pd.DataFrame(distance_functions[i](data.to_numpy(copy=True)))

					returns = initial_kmedoids_clustering(data.to_numpy(copy=True), mini, maxi, iterations, distances.to_numpy(copy=True))

					index = [x for x in range(mini, maxi)]
					cols = ["sil", "sil_clusters", "ic", "ic_clusters", "conn", "conn_clusters"]
					df = pd.DataFrame(columns = cols)
					for j in range(len(returns)):
						df[cols[j]] = returns[j]
					df.index = index
					df.to_csv(f'data/AMPds2/initial_clusterings/{hour}H_initial_{file}_{distance_functions[i].__name__}_seg_{k}.csv')

					# print("Starting", file)
					# data = pd.read_csv(f'data/AMPds2/z_{file}_segment_{k}.csv', index_col=0, parse_dates=True)
					# #data = data.apply(zscore, axis=1, result_type='expand')
					# distances = pd.DataFrame(distance_functions[i](data.to_numpy(copy=True)))

					# returns = initial_kmedoids_clustering(data.to_numpy(copy=True), mini, maxi, iterations, distances.to_numpy(copy=True))

					# index = [x for x in range(mini, maxi)]
					# cols = ["sil", "sil_clusters", "ic", "ic_clusters", "conn", "conn_clusters"]
					# df = pd.DataFrame(columns = cols)
					# for j in range(len(returns)):
					# 	df[cols[j]] = returns[j]
					# df.index = index
					# df.to_csv(f'data/AMPds2/initial_clusterings/z_initial_{file}_{distance_functions[i].__name__}_seg_{k}.csv')

def fastdtw_wrapper(data):
	return np.array([[fastdtw(data[x], data[y])[0] for x in range(len(data))]for y in range(len(data))])

def main():
	#ampds_multiple_distances()
	ampds()

if __name__ == '__main__':
	main()
