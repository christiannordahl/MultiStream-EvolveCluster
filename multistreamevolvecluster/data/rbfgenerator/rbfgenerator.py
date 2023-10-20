import pseudo_random_processes as prp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RBFGenerator():
	def __init__(self, n_clusters=[5], n_features=[2], n_cluster_range=[0], sample_random_state=None, model_random_state=None, kernel_radius=[0.07], kernel_radius_range=[0.0], drift_speed=[0.0], centroid_speeds=[None],event_frequency=10000,merge_split=False,create_delete=True, drift_delay=[0], n_streams=1, distributions=[None]):
		self.sample_random_state = sample_random_state
		self.model_random_state = model_random_state
		self.n_original_clusters = n_clusters
		self.n_clusters = [x for x in n_clusters]
		self.n_features = [x for x in n_features]
		self.n_current_clusters = [x for x in n_clusters]
		self.n_cluster_range = [x for x in n_cluster_range]
		self.kernel_radius = [x for x in kernel_radius]
		self.kernel_radius_range = [x for x in kernel_radius_range]
		self.drift_speed = [x for x in drift_speed]
		self.centroid_speeds = [x for x in centroid_speeds]
		self.centroids = None
		self.centroid_weights = None
		self.event_frequency=event_frequency
		self.event_calculator=0
		self.merge_split=merge_split
		self.create_delete=create_delete
		self.drift_delay=drift_delay
		### Addition for MultiStreams
		self.n_streams=n_streams
		### Addition for different distributions
		### len(distributions) should be = to n_streams
		self.distributions=distributions

		self._prepare_for_use()

	def move_centroids(self):
		for n in range(self.n_streams):
			for i in range(self.n_clusters[n]):
				if self.centroids[n][i] is not None:
					for j in range(self.n_features[n]):
						self.centroids[n][i].centre[j] += self.centroids[n][i].speed[j] * self.drift_speed[n]
						if (self.centroids[n][i].centre[j] > 1) or (self.centroids[n][i].centre[j] < 0):
							self.centroids[n][i].centre[j] = 1 if (self.centroids[n][i].centre[j] > 1) else 0
							self.centroids[n][i].speed[j] = -self.centroids[n][i].speed[j]

	def event(self):

		if self.merge_split and self.create_delete:
			_start = 0
			_stop = 4
		elif self.merge_split:
			_start = 0
			_stop = 2
		elif self.create_delete:
			_start = 2
			_stop = 4
		rand  = self._sample_random_state.randint(_start, _stop)

		if rand == 0:	# Merge
			self.merge()
		elif rand == 1:	# Split
			self.split()
		elif rand == 2:	# Create
			if self.n_current_clusters[0] < (self.n_original_clusters[0] + self.n_cluster_range[0]):
				self.create()
			else:
				self.delete()
		elif rand == 3:	# Delete
			if self.n_current_clusters[0] > (self.n_original_clusters[0] - self.n_cluster_range[0]):
				self.delete()
			else:
				self.create()

	def merge(self):
		pass
	def split(self):
		pass
	def create(self):
		for n in range(self.n_streams):
			self.centroids[n].append(Centroid())
			rand_centre = []
			for j in range(self.n_features[n]):
				rand_centre.append(self._sample_random_state.rand())
			self.centroids[n][-1].centre = rand_centre
			self.centroids[n][-1].class_label = len(self.centroids[n])-1
			#self.centroids[i].std_dev = model_random_state.rand()
			# Calculates a varying kernel radius, based on the kernel radius range variable.
			# If the radius range is 0, then std_dev is always same as kernel radius.
			self.centroids[n][-1].std_dev = self.kernel_radius[n] + (self._sample_random_state.randint(3)-1)*self.kernel_radius_range[n]
			self.centroid_weights[n].append(self._sample_random_state.rand())
			self.n_current_clusters[n] += 1
			self.n_clusters[n] += 1

		for n in range(self.n_streams):
			if self.centroid_speeds[n] is None:
				rand_speed = []
				norm_speed = 0.0
				for j in range(self.n_features[n]):
					rand_speed.append(self._sample_random_state.rand())
					norm_speed += rand_speed[j]*rand_speed[j]
				norm_speed = np.sqrt(norm_speed)
				for j in range(self.n_features[n]):
					rand_speed[j] /= norm_speed
				self.centroids[n][-1].speed = rand_speed
			else:
				self.centroids[n][-1].speed = self.centroid_speeds[n]


	def delete(self):
		i = prp.random_index_based_on_weights(self.centroid_weights[0], self._sample_random_state)
		while self.centroids[0][i] is None:
			i = prp.random_index_based_on_weights(self.centroid_weights[0], self._sample_random_state)

		for n in range(self.n_streams):
			self.n_current_clusters[n] -= 1
			self.centroids[n][i] = None
			self.centroid_weights[n][i] = 0.0


	def next_sample(self, batch_size=1):
		data = [np.zeros([batch_size, self.n_features[n]+1]) for n in range(self.n_streams)]
		if self.drift_delay > 0:
			self.drift_delay -= 1
		elif self.event_calculator == self.event_frequency:
			self.event()
			self.event_calculator = 0

		for j in range(batch_size):
			self.move_centroids()
			i = prp.random_index_based_on_weights(self.centroid_weights[0], self._sample_random_state)
			while self.centroids[0][i] is None:
				i = prp.random_index_based_on_weights(self.centroid_weights[0], self._sample_random_state)

			for n in range(self.n_streams):
				centroid_aux = self.centroids[n][i]
				att_vals = []
				magnitude = 0.0
				for k in range(self.n_features[n]):
					att_vals.append((self._sample_random_state.rand()*2.0)-1.0)
					magnitude += att_vals[k] * att_vals[k]
				magnitude = np.sqrt(magnitude)
				desired_mag = self._sample_random_state.normal() * centroid_aux.std_dev
				scale = desired_mag/magnitude
				for k in range(self.n_features[n]):
					data[n][j, k] = centroid_aux.centre[k] + att_vals[k] * scale
				data[n][j, self.n_features[n]] = centroid_aux.class_label
				self.event_calculator += 1

		for n in range(self.n_streams):
			data[n] = data[n].flatten()
		return data

	def _prepare_for_use(self):
		self._generate_centroids()
		self._sample_random_state = prp.check_random_state(self.sample_random_state)

	def _generate_centroids(self):
		model_random_state = prp.check_random_state(self.model_random_state)
		self.centroids = []
		self.centroid_weights = []
		for n in range(self.n_streams):
			self.centroids.append([])
			self.centroid_weights.append([])
			for i in range(self.n_clusters[n]):
				self.centroids[n].append(Centroid())
				rand_centre = []
				for j in range(self.n_features[n]):
					rand_centre.append(model_random_state.rand())
				self.centroids[n][i].centre = rand_centre
				self.centroids[n][i].class_label = i
				#self.centroids[i].std_dev = model_random_state.rand()
				# Calculates a varying kernel radius, based on the kernel radius range variable.
				# If the radius range is 0, then std_dev is always same as kernel radius.
				self.centroids[n][i].std_dev = self.kernel_radius[n] + (model_random_state.randint(3)-1)*self.kernel_radius_range[n]
				self.centroid_weights[n].append(model_random_state.rand())

		for n in range(self.n_streams):
			for i in range(self.n_clusters[n]):
				# Constant drift of centroids
				if self.centroid_speeds[n] is None:
					rand_speed = []
					norm_speed = 0.0
					for j in range(self.n_features[n]):
						rand_speed.append(model_random_state.rand()*2-1.0)

						norm_speed += rand_speed[j]*rand_speed[j]
					norm_speed = np.sqrt(norm_speed)
					for j in range(self.n_features[n]):
						rand_speed[j] /= norm_speed
					self.centroids[n][i].speed = rand_speed
				else:
					self.centroids[n][i].speed = self.centroid_speeds[n]


class Centroid():
	def __init__(self):
		self.centre = None
		self.std_dev = None
		self.class_label = None
		self.speed = None

def generate_aligned_data(number_of_instances=5000, n_clusters=[5], n_features=[2], n_cluster_range=[3], sample_random_state=99, model_random_state=50, kernel_radius=[0.02], kernel_radius_range=[0.005],drift_speed=[0.0001],centroid_speeds=[None],event_frequency=2500,merge_split=[False],create_delete=[True], drift_delay=0, n_streams=1):
	r = RBFGenerator(n_clusters, 
					n_features, 
					n_cluster_range, 
					sample_random_state, 
					model_random_state, 
					kernel_radius, 
					kernel_radius_range,
					drift_speed,
					centroid_speeds,
					event_frequency,
					merge_split,
					create_delete,
					drift_delay,
					n_streams)

	dfs = [[] for n in range(n_streams)]
	for i in range(number_of_instances):
		a = r.next_sample()
		for n in range(n_streams):
			dfs[n].append(a[n])

	for n in range(n_streams):
		dfs[n] = pd.DataFrame(dfs[n])
		dfs[n].columns = [*dfs[n].columns[:-1], 'cluster']
		dfs[n].cluster = dfs[n].cluster.astype(int)
	return dfs

def generate_misaligned_data(number_of_instances=5000, n_clusters=5, n_features=2, n_cluster_range=3, sample_random_state=99, model_random_state=50, kernel_radius=0.02, kernel_radius_range=0.005,drift_speed=0.0001,centroid_speeds=None,event_frequency=10000,merge_split=False,create_delete=True, drift_delay=0, ratio_misaligned=0.0):
	pass

def main():
	# n_streams=3
	# n_clusters=[5 for x in range(n_streams)]
	# n_features=[2 for x in range(n_streams)]
	# n_cluster_range=[3 for x in range(n_streams)]
	# kernel_radius=[0.02 for x in range(n_streams)]
	# kernel_radius_range=[0.005 for x in range(n_streams)]
	# drift_speed=[0.0001 for x in range(n_streams)]
	# centroid_speeds=[None for x in range(n_streams)]
	# event_frequency=2000
	# merge_split=False
	# create_delete=True
	# drift_delay=500
	# n_segments = 10

	# seeds = [1,6,2,12,29]
	# seeds = [x for x in range(100)]
	# # for seed in seeds:
	# # 	print(seed)
	# # 	number_of_instances = 10000
	# # 	data = generate_aligned_data(number_of_instances,
	# # 					n_clusters=n_clusters,
	# # 					n_features=n_features,
	# # 					n_cluster_range=n_cluster_range,
	# # 					sample_random_state=seed,
	# # 					model_random_state=seed,
	# # 					kernel_radius=kernel_radius,
	# # 					kernel_radius_range=kernel_radius_range,
	# # 					drift_speed=drift_speed,
	# # 					centroid_speeds=centroid_speeds,
	# # 					event_frequency=event_frequency,
	# # 					merge_split=merge_split,
	# # 					create_delete=create_delete,
	# # 					drift_delay=drift_delay,
	# # 					n_streams=n_streams)
		
	# # 	for n in range(n_streams):
	# # 		ddata = data[n]
	# # 		for i in range(n_segments):
	# # 			dat = ddata.iloc[(number_of_instances//n_segments)*i:(number_of_instances//n_segments)*(i+1),:]
	# # 			dat.to_csv('%d-dim/seed_%d_stream_%d_segment_%d_createdelete=True.csv'%(n_features[0],seed,n,i), index=False)

	# event_frequency=100001
	# create_delete=False
	# for seed in seeds:
	# 	print(seed)
	# 	number_of_instances = 10000
	# 	data = generate_aligned_data(number_of_instances,
	# 					n_clusters=n_clusters,
	# 					n_features=n_features,
	# 					n_cluster_range=n_cluster_range,
	# 					sample_random_state=seed,
	# 					model_random_state=seed,
	# 					kernel_radius=kernel_radius,
	# 					kernel_radius_range=kernel_radius_range,
	# 					drift_speed=drift_speed,
	# 					centroid_speeds=centroid_speeds,
	# 					event_frequency=event_frequency,
	# 					merge_split=merge_split,
	# 					create_delete=create_delete,
	# 					drift_delay=drift_delay,
	# 					n_streams=n_streams)
		
		
	# 	for n in range(n_streams):
	# 		ddata = data[n]
	# 		for i in range(n_segments):
	# 			dat = ddata.iloc[(number_of_instances//n_segments)*i:(number_of_instances//n_segments)*(i+1),:]
	# 			dat.to_csv('%d-dim/seed_%d_stream_%d_segment_%d_createdelete=False.csv'%(n_features[0],seed,n,i), index=False)

	n_streams=12
	n_clusters=[5 for x in range(n_streams)]
	n_features=[8 for x in range(n_streams)]
	n_cluster_range=[3 for x in range(n_streams)]
	kernel_radius=[0.02 for x in range(n_streams)]
	kernel_radius_range=[0.005 for x in range(n_streams)]
	drift_speed=[0.0001 for x in range(n_streams)]
	centroid_speeds=[None for x in range(n_streams)]
	event_frequency=2000
	merge_split=False
	create_delete=True
	drift_delay=500
	n_segments = 10

	seeds = [1,6,2,12,29]
	seeds = [x for x in range(100)]
	# for seed in seeds:
	# 	print(seed)
	# 	number_of_instances = 10000
	# 	data = generate_aligned_data(number_of_instances,
	# 					n_clusters=n_clusters,
	# 					n_features=n_features,
	# 					n_cluster_range=n_cluster_range,
	# 					sample_random_state=seed,
	# 					model_random_state=seed,
	# 					kernel_radius=kernel_radius,
	# 					kernel_radius_range=kernel_radius_range,
	# 					drift_speed=drift_speed,
	# 					centroid_speeds=centroid_speeds,
	# 					event_frequency=event_frequency,
	# 					merge_split=merge_split,
	# 					create_delete=create_delete,
	# 					drift_delay=drift_delay,
	# 					n_streams=n_streams)
		
	# 	for n in range(n_streams):
	# 		ddata = data[n]
	# 		for i in range(n_segments):
	# 			dat = ddata.iloc[(number_of_instances//n_segments)*i:(number_of_instances//n_segments)*(i+1),:]
	# 			dat.to_csv('%d-dim/seed_%d_stream_%d_segment_%d_createdelete=True.csv'%(n_features[0],seed,n,i), index=False)

	event_frequency=100001
	create_delete=False
	for seed in seeds:
		print(seed)
		number_of_instances = 10000
		data = generate_aligned_data(number_of_instances,
						n_clusters=n_clusters,
						n_features=n_features,
						n_cluster_range=n_cluster_range,
						sample_random_state=seed,
						model_random_state=seed,
						kernel_radius=kernel_radius,
						kernel_radius_range=kernel_radius_range,
						drift_speed=drift_speed,
						centroid_speeds=centroid_speeds,
						event_frequency=event_frequency,
						merge_split=merge_split,
						create_delete=create_delete,
						drift_delay=drift_delay,
						n_streams=n_streams)
		
		
		for n in range(n_streams):
			ddata = data[n]
			for i in range(n_segments):
				dat = ddata.iloc[(number_of_instances//n_segments)*i:(number_of_instances//n_segments)*(i+1),:]
				dat.to_csv('%d-dim/seed_%d_stream_%d_segment_%d_createdelete=False.csv'%(n_features[0],seed,n,i), index=False)


if __name__ == '__main__':
	main()
