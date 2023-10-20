import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial as spt
import pickle
import os
import json
import sys
import copy

from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import zscore
from functions import calculate_distances
from functions import kmedoids
from functions import IC_av
from matplotlib.patches import ConnectionPatch

import warnings

class EvolveCluster(object):
	def __init__(self, data, clusters, centroids, tau, distances=None, distance_function=None):
		"""
		Requires data to be initially clustered beforehand.
		"""
		self.data = data
		self.clusters = [clusters]
		self.centroids = [centroids]
		self.medoids = [[data[i] for i in self.centroids[-1]]]
		self.nr_increments = 0
		self.transitions = []
		self.tau = tau
		if distances is None:
			if distance_function is None:
				raise TypeError('Please submit a valid distance function to the EvolveCluster __init__ function.')
			else:
				self.distance_function = distance_function
				self.distances = None
		else:	# Legacy of old experiments.
			self.distances = [distances]

	def get_cluster_mapping(self):
		num_elements = sum([len(self.clusters[-1][x]) for x in self.clusters[-1]])
		mapping = [None for x in range(num_elements)]

		for cluster in self.clusters[-1]:
			for index in self.clusters[-1][cluster]:
				mapping[index] = cluster
		return mapping

	def get_specific_cluster_mapping(self, segment):
		num_elements = sum([len(self.clusters[segment][x]) for x in self.clusters[segment]])
		mapping = [None for x in range(num_elements)]

		for cluster in self.clusters[segment]:
			for index in self.clusters[segment][cluster]:
				mapping[index] = cluster
		return mapping

	def homogenity(self, mean=True):
		if len(np.argwhere(np.isnan(self.distances))) > 1:
			print(self.data[np.argwhere(np.isnan(self.distances))])
			print(self.nr_increments)
		#clusters = [None for x in range(len(self.data))]
		clusters = [None for x in range(len(self.distances))]
		for i in range(len(self.centroids[self.nr_increments])):
			for index in self.clusters[self.nr_increments][i]:
				clusters[index] = self.centroids[self.nr_increments][i]

		#print(self.nr_increments)
		#print(np.argwhere(np.isnan(self.distances)))
		sil_scores = silhouette_samples(self.distances, clusters, metric='precomputed')
		
		div_scores = [[] for x in range(len(self.centroids[self.nr_increments]))]
		for key in self.clusters[self.nr_increments]:
			for index in self.clusters[self.nr_increments][key]:
				div_scores[key].append(sil_scores[index])

		if mean == True:
			for i in range(len(div_scores)):
				div_scores[i] = np.mean(np.array(div_scores[i]))

		return div_scores

	def split(self, data):
		if(len(self.centroids[self.nr_increments]) > 1):
			total_hom_score = self.homogenity(False)
			total_hom_score = np.mean([y for x in total_hom_score for y in x])
			homogenity_scores = self.homogenity()
			indices_low_scores = sorted(range(len(homogenity_scores)), key = lambda sub: homogenity_scores[sub])[:]

			i = 0
			while i < len(indices_low_scores):
				if len(self.clusters[self.nr_increments][indices_low_scores[i]]) > 1:
					old_medoids = copy.deepcopy(self.medoids[self.nr_increments])
					old_clusters = copy.deepcopy(self.clusters[self.nr_increments])
					old_centroids = copy.deepcopy(self.centroids[self.nr_increments])

					distances = self.distances
					indices = self.clusters[self.nr_increments][indices_low_scores[i]]
					#mask = [x in indices for x in range(len(self.data))]
					mask = [x in indices for x in range(len(distances))]
					mask2 = np.array([[x & y for x in mask] for y in mask])
					index_converter = [x for x in range(len(mask)) if mask[x] is True]

					distances = distances[mask2]
					distances = np.reshape(distances, (-2,int(len(distances)**0.5)))
					centroids = np.where(distances == np.amax(distances))
					centroids = [centroids[0][0],centroids[-1][0]]

					C, new_centroids = kmedoids(distances, 2, centroids)

					cluster_indices = np.array([index_converter[x] for x in C[0]])
					self.clusters[self.nr_increments][indices_low_scores[i]] = cluster_indices
					cluster_indices = [index_converter[x] for x in C[1]]
					self.clusters[self.nr_increments][len(self.clusters[self.nr_increments])] = cluster_indices
					self.centroids[self.nr_increments][indices_low_scores[i]] = index_converter[new_centroids[0]]
					self.centroids[self.nr_increments].append(index_converter[new_centroids[1]])

					C, centroids = kmedoids(self.distances, 
											len(self.centroids[self.nr_increments]), 
											self.centroids[self.nr_increments])
					self.clusters[self.nr_increments] = C
					self.centroids[self.nr_increments] = centroids
					self.medoids[self.nr_increments] = [data[i] for i in centroids]

					new_homogenity_scores = self.homogenity(False)
					new_hom_score = np.mean([y for x in new_homogenity_scores for y in x])

					if new_hom_score > (total_hom_score + abs(total_hom_score*self.tau)):
						self.transitions[self.nr_increments-1][indices_low_scores[i]] = [indices_low_scores[i], len(self.clusters[self.nr_increments])-1]
						return True

					self.clusters[self.nr_increments] = old_clusters
					self.centroids[self.nr_increments] = old_centroids
					self.medoids[self.nr_increments] = old_medoids
				i += 1

		return False

	def cluster_stream(self, data):
		i = 0
		for seg in data:
			i+=1
			print(i)
			warnings.simplefilter("ignore")
			self.cluster_method(seg)
			while(self.split(seg)):
				#print('Did split')
				pass
			self.distances = None
	def cluster_segment(self, data):
		self.cluster_method(data)
		while(self.split(data)):
			#print('Did split')
			pass
		self.distances = None

	def cluster_method(self, data):
		"""
		Requires self-implementation depending on what clustering method
		you choose. Receives data as a parameter and has to return a list
		containing a list of corresponding cluster values per element and 
		a list of centroids. Below is an implementation using k-medoids.
		"""
		centroids = self.centroids[self.nr_increments]
		num_clusters = len(centroids)
		for medoid in self.medoids[self.nr_increments]:
			data = np.concatenate((data, [medoid]))#.iloc[centroid,:].to_numpy(copy=True)]))	# Behöver ta self.medoids[] istället för self.data[centroid]
		centroids = [x for x in range(len(data)-num_clusters, len(data))]

		D = self.distance_function(data)
		data = data[:-num_clusters]
		#data = None 	# Deallocate memory
		self.data = None # Deallocate memory

		C = {}
		C, D, centroids = self.initial_partiton(C, D, centroids, num_clusters)

		transitions = {}
		j = 0
		for i in range(num_clusters):
			transitions[i] = None
			if centroids[i] is not None:
				transitions[i] = [j]
				j += 1
		for i in range(num_clusters-1, -1, -1):
			if centroids[i] == None:
				del centroids[i]
				num_clusters -= 1
		self.transitions.append(transitions)

		C, centroids = kmedoids(D, num_clusters, centroids)
		
		self.distances = D
		self.centroids.append(centroids)
		self.medoids.append([data[i] for i in centroids])
		self.clusters.append(C)
		self.data = data # Memory deallocation
		self.nr_increments += 1

	def initial_partiton(self, C, D, medoids, num_clusters):
		J = np.argmin(D[:,medoids], axis=1)
		for kappa in range(num_clusters):
			C[kappa] = np.where(J==kappa)[0]

		D = D[:-num_clusters,:-num_clusters]
		for kappa in range(num_clusters):
			toDelete = [i for i in range(len(C[kappa])) if C[kappa][i] == medoids[kappa]]	# List comprehension of the below.
			C[kappa] = np.delete(C[kappa],toDelete)
			# toDelete = []
			# for i in range(len(C[kappa])):
			#	if C[kappa][i] == medoids[kappa]:
			#		#C[kappa] = np.delete(C[kappa], i)	# When a deletion occurs the range extends for too long, causing IndexError in the C[kappa] array
			#		toDelete.append(i)					# This corrects the behavior as we first search through, then we delete when we identified all i's
			# C[kappa] = np.delete(C[kappa],toDelete)
					

		
		warnings.simplefilter("ignore")
		for kappa in range(num_clusters):
			J = np.mean(D[np.ix_(C[kappa],C[kappa])], axis=1)
			if J.size == 0:
				del C[kappa]
				medoids[kappa] = None
			else:
				j = np.argmin(J)
				medoids[kappa] = C[kappa][j]

		return C, D, medoids
