import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
import json
import matplotlib.pyplot as plt
import matplotlib.colors
import sys

from MultiStreamEvolveCluster import MultiStreamEvolveCluster

from functions import pairwise_euclidean, IC_av
from fastdtw import fastdtw

import pickle

def figures_and_tables_single(seeds, n_streams, n_segments, createdelete=False):
	eca_F, eca_J, eca_S = [],[],[]
	eca_files = []
	eca_filter_files = []
	for seed in seeds:
		print(seed)
		plot_files = []
		top_labels=[f"S$^{x}_{{n}}$" for x in range(n_segments)]
		for stream in range(n_streams):
			plot_files.append([pd.read_csv(f'2-dim/seed_{seed}_stream_{stream}_segment_{segment}_createdelete={createdelete}.csv') for segment in range(n_segments)])

		plot_combined(plot_files, f'seed_{seed}_n_streams_{n_streams}_createdelete_{createdelete}.pdf', top_labels)

	# for i in range(5):
	# 	df = eca_plot_files[i]
	# 	df['cluster'] = df['cluster_ids']
	# 	df['_id'] = df.index
	# 	#df = df.merge(basefile, on='_id', how='inner')
	# 	eca_F.append(cluster_wise_f_measure(df))
	# 	eca_J.append(cluster_wise_jaccard(df))
	# 	D = calculate_distances(df[['0','1']].to_numpy(copy=True))
	# 	try:
	# 		eca_S.append(silhouette_score(D, df["cluster"], metric="precomputed"))
	# 	except ValueError:
	# 		eca_S.append(-11)
	# data = [eca_F, eca_J, eca_S]
	# table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=top_labels)
	# table_df.to_latex(open("%s.tex"%(seed_dim), "w"), float_format="%.3f", caption="asd", label="tab:s1-cont-eca-single", escape=False)


def plot_combined(data, output, top_labels):
	colors = [plt.cm.tab20(x) for x in range(20)]
	cmap = matplotlib.colors.ListedColormap(colors)
	figsize = (24,2.6)
	
	fig, axs = plt.subplots(len(data),len(data[0]),figsize=figsize, sharex=True, sharey=True)
	for i in range(len(data)):
		for j in range(len(data[i])):
			if i == 0:
				axs[i][j].set_title(top_labels[j], fontsize=20)
			for k in sorted(data[i][j]['cluster'].unique()):
				axs[i][j].scatter(data[i][j].loc[data[i][j]['cluster'] == k]['0'],
							   data[i][j].loc[data[i][j]['cluster'] == k]['1'],
							   color=colors[k], s=4, alpha=0.5)
			#axs[j].scatter(data[i][j]['0'], 
			#			data[i][j]['1'], 
			#			c=data[i][j]['cluster_ids'], cmap='tab20', s=4,alpha=0.5)

		plt.ylim(-0.1,1.1)
		plt.xlim(-0.1,1.1)
	plt.tight_layout()
	fig.savefig(output, bbox_inches='tight')
	plt.close()


def plot(data, clusters, output):
	fig, axs = plt.subplots(1,len(data),figsize=(20,3), sharex=True, sharey=True)
	for i in range(len(data)):
		axs[i].scatter(data[i]['attributes'].map(lambda x: x[0]), 
						data[i]['attributes'].map(lambda x: x[1]), 
						c=data[i]['cluster'], cmap='tab20')
		# for j in range(len(clusters[i])):
		# 	axs[i].scatter(clusters[i][j][0],clusters[i][j][1],color='black')
		#for cluster in clusters::
		#	for index in self.clusters[i][key]:
		#		axs[i].scatter(self.data[i][index][0], self.data[i][index][1], color=colors[key], s=30)
		#	axs[i].scatter(self.data[i][self.centroids[i][key]][0], self.data[i][self.centroids[i][key]][1], color=colors[key%len(colors)],edgecolors='black', linewidth=1, s=30)

		plt.ylim(-0.1,1.1)
		plt.xlim(-0.1,1.1)
	plt.tight_layout()
	fig.savefig(output, bbox_inches='tight')
	plt.close()

def calculate_distances(data):
    return pairwise_distances(data,data)

def cluster_wise_jaccard(data):
	sum = 0
	unique_clusters_predicted = sorted(data['cluster'].unique())
	for cluster in unique_clusters_predicted:
		predicted_cluster = data[data['cluster'] == cluster]
		true_cluster = cluster_with_max_shared(predicted_cluster, data)
		sum += jaccard_measure(set(predicted_cluster['_id']), set(true_cluster['_id']))

	length = len(unique_clusters_predicted)
	return (sum / length)

def jaccard_measure(pred, true):
	value = (len(pred & true)) / (len(true) + len(pred) - len(true & pred))
	return value

def cluster_wise_f_measure(data, cluster_column='cluster'):
    sum = 0
    # unique_clusters_predicted = sorted(predicted["cluster"].unique())

    # for cluster in unique_clusters_predicted:
    #     predicted_cluster = predicted[predicted["cluster"] == cluster]
    #     true_cluster = cluster_with_max_shared_experts(predicted_cluster, true)
    #     sum = sum + f_measure(set(predicted_cluster["_id"]), set(true_cluster["_id"]))
    unique_clusters_predicted = sorted(data[cluster_column].unique())
    for cluster in unique_clusters_predicted:
    	predicted_cluster = data[data[cluster_column] == cluster]
    	true_cluster = cluster_with_max_shared(predicted_cluster, data)
    	sum += f_measure(set(predicted_cluster['_id']), set(true_cluster['_id']))

    length = len(unique_clusters_predicted)
    return (sum / length)

def f_measure(pred, true):
    value = (2 * len(pred & true)) / (len(true) + len(pred))
    return value

def cluster_with_max_shared(predicted_cluster, data, actual_cluster_column='actual_cluster'):
	predicted = set(predicted_cluster['_id'])
	true = data[actual_cluster_column].unique()
	max_cluster = 0
	max_session = 0

	for label in true:
		cluster = data[data[actual_cluster_column] == label]
		same_cluster = len(predicted & set(cluster['_id']))
		if same_cluster > max_cluster:
			max_cluster = same_cluster
			max_session = label
	return data[data[actual_cluster_column] == max_session]
	
	# predicted_experts = set(predicted_cluster["_id"])
	# true_sessions = sorted(experts["cluster"].unique())
	# max_cluster = 0
	# max_session = 0
	# for disease in true_sessions:
	# 	cluster = experts[experts["cluster"] == disease]
	# 	same_experts = len(predicted_experts & set(cluster["_id"]))
	# 	if same_experts > max_cluster:
	# 		max_cluster = same_experts
	# 		max_session = disease

	# return experts[experts["cluster"] == max_session]

def s_new_indiv_metrics(dataset, length):
	data = [pd.read_csv('data/s1new/%s/results/%d.csv'%(dataset, i), index_col=0) for i in range(length)]
	F = []
	J = []
	S = []
	for i in range(len(data)):
		f = cluster_wise_f_measure(data[i])
		F.append(f)

		j = cluster_wise_jaccard(data[i])
		J.append(j)

		D = calculate_distances(data[i].iloc[:,:2].to_numpy(copy=True))
		S.append(silhouette_score(D, data[i]["cluster"], metric="precomputed"))
	
	data = [F, J, S]
	table_df = pd.DataFrame(data, index=["F$_1$", "Jaccard", "Silhouette"], columns=["D$_0$","D$_1$","D$_2$","D$_3$","D$_4$"])
	table_df.to_latex(open("tables/s1/%s.tex"%(dataset), "w"), float_format="%.5f", caption="", label="")

def max_shared_cluster(predicted_cluster, data, true_cluster_column='actual_cluster'):
	predicted = set(predicted_cluster.index.tolist())
	true_clusters = data[true_cluster_column].unique()
	max_cluster = 0
	max_session = 0

	for label in true_clusters:
		cluster = data[data[true_cluster_column] == label]
		same_cluster = len(predicted & set(cluster.index.tolist()))
		if same_cluster > max_cluster:
			max_cluster = same_cluster
			max_session = label
	return data[data[true_cluster_column] == max_session]

def cluster_wise_f(data, cluster_column='cluster', true_cluster_column='actual_cluster'):
    sum = 0
    unique_clusters_predicted = sorted(data[cluster_column].unique())
    for cluster in unique_clusters_predicted:
    	predicted_cluster = data[data[cluster_column] == cluster]
    	true_cluster = max_shared_cluster(predicted_cluster, data, true_cluster_column=true_cluster_column)
    	sum += f_measure(set(predicted_cluster.index.tolist()), set(true_cluster.index.tolist()))

    length = len(unique_clusters_predicted)
    return (sum / length)

def cluster_wise_j(data, cluster_column='cluster', true_cluster_column='actual_cluster'):
	sum = 0
	unique_clusters_predicted = sorted(data[cluster_column].unique())
	for cluster in unique_clusters_predicted:
		predicted_cluster = data[data[cluster_column] == cluster]
		true_cluster = max_shared_cluster(predicted_cluster, data, true_cluster_column=true_cluster_column)
		sum += jaccard_measure(set(predicted_cluster.index.tolist()), set(true_cluster.index.tolist()))

	length = len(unique_clusters_predicted)
	return (sum / length)

def consensus_metrics_rbf(n_segments=10, n_streams=3, dimension=2, distance_function=pairwise_euclidean, table_path='results/rbfgenerator/tables/'):
	total_dimension=dimension*n_streams
	seeds = [72,75,82,83,91,92,94]

	for seed in seeds:
		### Data grabbing
		data = []
		for stream in range(n_streams):
			data.append([pd.read_csv(f'data/rbfgenerator/{dimension}-dim/seed_{seed}_stream_{stream}_segment_{segment}_createdelete=False.csv') for segment in range(n_segments)])
		
		combined_data = []
		for segment in range(n_segments):
			combined_data.append(data[stream][segment].copy())
			for stream in range(1, n_streams):
				for i in range(dimension):
					combined_data[segment][f'{i+stream*dimension}'] = data[stream][segment][f'{i}']
			cols = combined_data[segment].columns.tolist()
			cols = cols[:dimension]+cols[(dimension+1):]+[cols[dimension]]	# Swaps true cluster to end
			combined_data[segment] = combined_data[segment][cols]
			combined_data[segment].rename(columns={'cluster': 'true_cluster'}, inplace=True)

		### Cluster grabbing
		MSEC = pickle.load(open(f'results/rbfgenerator/{dimension}-dim/MSEC_seed_{seed}.bin','rb'))
		for stream in range(n_streams):
			for segment in range(n_segments):
				clusters = MSEC.modules[stream].get_specific_cluster_mapping(segment)
				data[stream][segment].rename(columns={'cluster': 'true_cluster'}, inplace=True)
				data[stream][segment]['cluster'] = clusters

		EC = pickle.load(open(f'results/rbfgenerator/{dimension}-dim/EC_seed_{seed}.bin', 'rb'))
		for segment in range(n_segments):
			# Extract clusterings from EC object
			clustering = [-1 for x in range(len(MSEC.consensus_clusters_MV[segment]))]
			for cluster in EC.clusters[segment]:
				for index in EC.clusters[segment][cluster]:
					clustering[index] = cluster

			combined_data[segment]['cluster_MV'] = MSEC.consensus_clusters_MV[segment]
			combined_data[segment]['cluster_HGP'] = MSEC.consensus_clusters_HGP[segment]
			combined_data[segment]['cluster_MC'] = MSEC.consensus_clusters_MC[segment]
			combined_data[segment]['cluster_baseline'] = clustering

		### Distance measures
		distances = []
		combined_distances = []
		for stream in range(n_streams):
			distances.append([])
			for segment in range(n_segments):
				distances[stream].append(distance_function(data[stream][segment].iloc[:,:dimension].to_numpy(copy=True)))
		for segment in range(n_segments):
			combined_data.append(distance_function(combined_data[segment].iloc[:,:total_dimension].to_numpy(copy=True)))

		# Individual streams
		cols = columns=[f'$S^{{{x}}}$' for x in range(1,11)]
		evaluation_data = []
		for stream in range(n_streams):
			F, J, SI, ARI, AMI, CH, DB, NMI, n_clusters = [], [], [], [], [], [], [], [], []
			for segment in range(n_segments):
				F.append(cluster_wise_f(data[stream][segment], cluster_column='cluster', true_cluster_column='true_cluster'))
				J.append(cluster_wise_j(data[stream][segment], cluster_column='cluster', true_cluster_column='true_cluster'))
				D = distance_function(data[stream][segment].iloc[:,:dimension].to_numpy(copy=True))
				try:
					SI.append(silhouette_score(D, data[stream][segment]['cluster'], metric='precomputed'))
				except:
					SI.append(-11)
				ARI.append(adjusted_rand_score(data[stream][segment]['true_cluster'], data[stream][segment]['cluster']))
				AMI.append(adjusted_mutual_info_score(data[stream][segment]['true_cluster'], data[stream][segment]['cluster']))
				try:
					CH.append(calinski_harabasz_score(data[stream][segment].iloc[:,:dimension], data[stream][segment]['true_cluster']))
				except:
					CH.append(-11)
				try:
					DB.append(davies_bouldin_score(data[stream][segment].iloc[:,:dimension], data[stream][segment]['true_cluster']))
				except:
					DB.append(-11)
				NMI.append(normalized_mutual_info_score(data[stream][segment]['true_cluster'], data[stream][segment]['cluster']))
				n_clusters.append(len(data[stream][segment]['cluster'].unique()))
			evaluation_data.append(F)
			evaluation_data.append(J)
			evaluation_data.append(SI)
			evaluation_data.append(ARI)
			evaluation_data.append(AMI)
			evaluation_data.append(CH)
			evaluation_data.append(DB)
			evaluation_data.append(NMI)
			evaluation_data.append(n_clusters)

		measures = ['F$_1$','J','SI', 'ARI', 'AMI', 'CH', 'DB', 'NMI', 'num clusters']
		indices = pd.MultiIndex.from_tuples([(f'$S_{{{stream}}}$', measure) for stream in range(n_streams) for measure in measures])
		df_scores = pd.DataFrame(evaluation_data, columns=cols, index=indices)
		s = df_scores.style
		with open(f'{table_path}{dimension}-dim_scores_{seed}.tex', 'w') as f:
			f.write(s.to_latex(hrules=True))

		# Consensus and baseline
		evaluation_data = []
		cluster_cols = ['cluster_MV', 'cluster_HGP', 'cluster_MC', 'cluster_baseline']
		for col in cluster_cols:
			F, J, SI, ARI, AMI, CH, DB, NMI, n_clusters = [], [], [], [], [], [], [], [], []
			n_clusters = []
			for segment in range(n_segments):
				F.append(cluster_wise_f(combined_data[segment], cluster_column=col, true_cluster_column='true_cluster'))
				J.append(cluster_wise_j(combined_data[segment], cluster_column=col, true_cluster_column='true_cluster'))
				D = distance_function(combined_data[segment].iloc[:,:total_dimension].to_numpy(copy=True))
				try:
					SI.append(silhouette_score(D, combined_data[segment][col], metric='precomputed'))
				except:
					SI.append(-11)
				ARI.append(adjusted_rand_score(combined_data[segment]['true_cluster'], combined_data[segment][col]))
				AMI.append(adjusted_mutual_info_score(combined_data[segment]['true_cluster'], combined_data[segment][col]))
				try:
					CH.append(calinski_harabasz_score(combined_data[segment].iloc[:,:total_dimension], combined_data[segment][col]))
				except:
					CH.append(-11)
				try:
					DB.append(davies_bouldin_score(combined_data[segment].iloc[:,:total_dimension], combined_data[segment][col]))
				except:
					DB.append(-11)
				NMI.append(normalized_mutual_info_score(combined_data[segment]['true_cluster'], combined_data[segment][col]))
				n_clusters.append(len(combined_data[segment][col].unique()))
			evaluation_data.append(F)
			evaluation_data.append(J)
			evaluation_data.append(SI)
			evaluation_data.append(ARI)
			evaluation_data.append(AMI)
			evaluation_data.append(CH)
			evaluation_data.append(DB)
			evaluation_data.append(NMI)
			evaluation_data.append(n_clusters)

		methods = ['Majority Vote', 'Hypergraph Partitioning', 'Markov Clustering', 'Baseline']
		methods = ['MV', 'HGP', 'MC', 'Base']
		indices = pd.MultiIndex.from_tuples([(f'{method}', measure) for method in methods for measure in measures])
		df_scores = pd.DataFrame(evaluation_data, columns=cols, index=indices)
		s = df_scores.style
		with open(f'{table_path}{dimension}-dim_total_scores_{seed}.tex', 'w') as f:
			f.write(s.to_latex(hrules=True))

# Needs to be adapted to the new format.
def consensus_metrics_ampds_new(n_segments=12, dimension=24, n_streams=4, table_path='results/AMPds2/tables/'):
	total_dimension=24
	streams = ['elec', 'water', 'gas', 'weather']
	best_initial = {'elec':[4,4,'pairwise_euclidean', pairwise_euclidean],
                    'water':[8,3,'pairwise_euclidean',pairwise_euclidean],
                    'gas':[6,3,'pairwise_euclidean',pairwise_euclidean],
                    'weather':[4,4,'canberra',canberra_wrapper]}

	data = []
	for stream in streams:
		data.append([pd.read_csv(f'data/AMPds2/{best_initial[stream][0]}H_{stream}_segment_{segment}.csv').iloc[:,1:] for segment in range(n_segments)])
	
	combined_data = [pd.read_csv(f'data/AMPds2/4H_all_segment_{segment}.csv').iloc[:,1:] for segment in range(n_segments)]

	### Cluster grabbing
	MSEC = pickle.load(open('results/AMPds2/MSEC.bin','rb'))
	for stream in range(n_streams):
		for segment in range(n_segments):
			clusters = MSEC.modules[stream].get_specific_cluster_mapping(segment)
			data[stream][segment]['cluster'] = clusters

	EC = pickle.load(open(f'results/AMPds2/EC_4H.bin', 'rb'))
	for segment in range(n_segments):
		# Extract clusterings from EC object
		print(len(MSEC.consensus_clusters_MV), segment)
		clustering = [-1 for x in range(len(MSEC.consensus_clusters_MV[segment]))]
		for cluster in EC.clusters[segment]:
			for index in EC.clusters[segment][cluster]:
				clustering[index] = cluster

		combined_data[segment]['cluster_MV'] = MSEC.consensus_clusters_MV[segment]
		combined_data[segment]['cluster_HGP'] = MSEC.consensus_clusters_HGP[segment]
		combined_data[segment]['cluster_MC'] = MSEC.consensus_clusters_MC[segment]
		combined_data[segment]['cluster_baseline'] = clustering

	### Distance measures
	distances = []
	combined_distances = []
	for stream in range(n_streams):
		distances.append([])
		for segment in range(n_segments):
			distances[stream].append(best_initial[streams[stream]][3](data[stream][segment].iloc[:,:dimension].to_numpy(copy=True)))
	for segment in range(n_segments):
		combined_distances.append(pairwise_euclidean(combined_data[segment].iloc[:,:total_dimension].to_numpy(copy=True)))

	# Individual streams
	cols = columns=[f'$S^{{{x}}}$' for x in range(1,13)]
	evaluation_data = []
	for stream in range(n_streams):
		IC, SI, n_clusters = [], [], [] 
		for segment in range(n_segments):
			D = best_initial[streams[stream]][3](data[stream][segment].iloc[:,:dimension].to_numpy(copy=True))
			try:
				SI.append(silhouette_score(D, data[stream][segment]['cluster'], metric='precomputed'))
			except:
				SI.append(-11)
			IC.append(IC_av(pd.DataFrame(D), data[stream][segment]['cluster'])[0])
			n_clusters.append(len(data[stream][segment]['cluster'].unique()))
		evaluation_data.append(IC)
		evaluation_data.append(SI)
		evaluation_data.append(n_clusters)

	measures = ['IC-av','SI', 'num clusters']
	indices = pd.MultiIndex.from_tuples([(f'$S_{{{streams[stream]}}}$', measure) for stream in range(n_streams) for measure in measures])
	df_scores = pd.DataFrame(evaluation_data, columns=cols, index=indices)
	s = df_scores.style
	s.format(precision=2)
	with open(f'{table_path}streams.tex', 'w') as f:
		f.write(s.to_latex(hrules=True))

	# Consensus and baseline
	evaluation_data = []
	cluster_cols = ['cluster_MV', 'cluster_HGP', 'cluster_MC', 'cluster_baseline']
	for col in cluster_cols:
		IC, SI, n_clusters = [], [], []
		n_clusters = []
		for segment in range(n_segments):
			D = pairwise_euclidean(combined_data[segment].iloc[:,:total_dimension].to_numpy(copy=True))
			try:
				SI.append(silhouette_score(D, combined_data[segment][col], metric='precomputed'))
			except:
				SI.append(-11)
			IC.append(IC_av(pd.DataFrame(D), combined_data[segment][col])[0])
			n_clusters.append(len(combined_data[segment][col].unique()))
		evaluation_data.append(IC)
		evaluation_data.append(SI)
		evaluation_data.append(n_clusters)

	methods = ['Majority Vote', 'Hypergraph Partitioning', 'Markov Clustering', 'Baseline']
	methods = ['MV', 'HGP', 'MC', 'Base']
	indices = pd.MultiIndex.from_tuples([(f'{method}', measure) for method in methods for measure in measures])
	df_scores = pd.DataFrame(evaluation_data, columns=cols, index=indices)
	s = df_scores.style
	s.format(precision=2)
	with open(f'{table_path}total.tex', 'w') as f:
		f.write(s.to_latex(hrules=True))


# Needs to be adapted to the new format.
def consensus_metrics_ampds_subset(n_segments=12, dimension=24, n_streams=4, table_path='results/AMPds2/tables/'):
	total_dimension=dimension*n_streams
	streams = ['elec', 'water', 'gas', 'weather']
	best_initial = {'elec':[4,4,'pairwise_euclidean', pairwise_euclidean],
                    'water':[8,3,'pairwise_euclidean',pairwise_euclidean],
                    'gas':[6,3,'pairwise_euclidean',pairwise_euclidean],
                    'weather':[4,4,'canberra',canberra_wrapper]}

	data = []
	for stream in streams:
		data.append([pd.read_csv(f'data/AMPds2/{best_initial[stream][0]}H_{stream}_segment_{segment}.csv').iloc[:,1:] for segment in range(n_segments)])
	
	combined_data = [pd.read_csv(f'data/AMPds2/all_segment_{segment}.csv').iloc[:,1:] for segment in range(n_segments)]

	msec_files = ['results/AMPds2/MSEC_without_weather_and_gas.bin','results/AMPds2/MSEC_without_weather.bin', 'results/AMPds2/MSEC_without_water_and_weather.bin','results/AMPds2/MSEC_without_water.bin']
	msec_files = ['results/AMPds2/thresh=0.6_MSEC_without_weather_and_gas.bin','results/AMPds2/thresh=0.6_MSEC_without_weather.bin', 'results/AMPds2/thresh=0.6_MSEC_without_water_and_weather.bin','results/AMPds2/thresh=0.6_MSEC_without_water.bin']
	msec_files = ['results/AMPds2/thresh=0.8_MSEC_without_weather_and_gas.bin','results/AMPds2/thresh=0.8_MSEC_without_weather.bin', 'results/AMPds2/thresh=0.8_MSEC_without_water_and_weather.bin','results/AMPds2/thresh=0.8_MSEC_without_water.bin']
	n_stream = [2,3,2,3]	# This is for separate streams

	msec_files = ['results/AMPds2/thresh=0.6_MSEC.bin', 'results/AMPds2/thresh=0.8_MSEC.bin']
	n_stream = [4,4] # This is for thresh_0.6/0.8.MSEC.bin

	i=0
	for msec_file, n_streams in zip(msec_files, n_stream):
		i+=1
		if i == 3:
			streams = ['elec', 'gas', 'weather', 'water']
		### Cluster grabbing
		MSEC = pickle.load(open(msec_file,'rb'))
		for stream in range(n_streams):
			for segment in range(n_segments):
				clusters = MSEC.modules[stream].get_specific_cluster_mapping(segment)
				data[stream][segment]['cluster'] = clusters

		EC = pickle.load(open(f'results/AMPds2/EC_pairwise_euclidean.bin', 'rb'))	# Behöver uppdateras när vi kör på riktigt
		for segment in range(n_segments):
			# Extract clusterings from EC object
			clustering = [-1 for x in range(len(MSEC.consensus_clusters_MV[segment]))]
			for cluster in EC.clusters[segment]:
				for index in EC.clusters[segment][cluster]:
					clustering[index] = cluster

			combined_data[segment]['cluster_MV'] = MSEC.consensus_clusters_MV[segment]
			combined_data[segment]['cluster_HGP'] = MSEC.consensus_clusters_HGP[segment]
			combined_data[segment]['cluster_MC'] = MSEC.consensus_clusters_MC[segment]
			combined_data[segment]['cluster_baseline'] = clustering

		### Distance measures
		distances = []
		combined_distances = []
		for stream in range(n_streams):
			distances.append([])
			for segment in range(n_segments):
				distances[stream].append(best_initial[streams[stream]][3](data[stream][segment].iloc[:,:dimension].to_numpy(copy=True)))
		for segment in range(n_segments):
			combined_distances.append(pairwise_euclidean(combined_data[segment].iloc[:,:total_dimension].to_numpy(copy=True)))

		# Individual streams
		cols = columns=[f'$S^{{{x}}}$' for x in range(1,13)]
		evaluation_data = []
		for stream in range(n_streams):
			IC, SI, n_clusters = [], [], [] 
			for segment in range(n_segments):
				D = best_initial[streams[stream]][3](data[stream][segment].iloc[:,:dimension].to_numpy(copy=True))
				try:
					SI.append(silhouette_score(D, data[stream][segment]['cluster'], metric='precomputed'))
				except:
					SI.append(-11)
				IC.append(IC_av(pd.DataFrame(D), data[stream][segment]['cluster'])[0])
				n_clusters.append(len(data[stream][segment]['cluster'].unique()))
			evaluation_data.append(IC)
			evaluation_data.append(SI)
			evaluation_data.append(n_clusters)

		measures = ['IC-av','SI', 'num clusters']
		indices = pd.MultiIndex.from_tuples([(f'$S_{{{streams[stream]}}}$', measure) for stream in range(n_streams) for measure in measures])
		df_scores = pd.DataFrame(evaluation_data, columns=cols, index=indices)
		s = df_scores.style
		s.format(precision=2)
		fpath_string = msec_file.split('/')[-1].split('.')[0]
		fpath_string = msec_file.split('/')[-1]
		with open(f'{table_path}{fpath_string}_streams.tex', 'w') as f:
			f.write(s.to_latex(hrules=True))

		# Consensus and baseline
		evaluation_data = []
		cluster_cols = ['cluster_MV', 'cluster_HGP', 'cluster_MC', 'cluster_baseline']
		for col in cluster_cols:
			IC, SI, n_clusters = [], [], []
			n_clusters = []
			for segment in range(n_segments):
				D = pairwise_euclidean(combined_data[segment].iloc[:,:total_dimension].to_numpy(copy=True))
				try:
					SI.append(silhouette_score(D, combined_data[segment][col], metric='precomputed'))
				except:
					SI.append(-11)
				IC.append(IC_av(pd.DataFrame(D), combined_data[segment][col])[0])
				n_clusters.append(len(combined_data[segment][col].unique()))
			evaluation_data.append(IC)
			evaluation_data.append(SI)
			evaluation_data.append(n_clusters)

		methods = ['Majority Vote', 'Hypergraph Partitioning', 'Markov Clustering', 'Baseline']
		methods = ['MV', 'HGP', 'MC', 'Base']
		indices = pd.MultiIndex.from_tuples([(f'{method}', measure) for method in methods for measure in measures])
		df_scores = pd.DataFrame(evaluation_data, columns=cols, index=indices)
		s = df_scores.style
		s.format(precision=2)
		fpath_string = msec_file.split('/')[-1].split('.')[0]
		fpath_string = msec_file.split('/')[-1]
		with open(f'{table_path}_{fpath_string}_total.tex', 'w') as f:
			f.write(s.to_latex(hrules=True))

def consensus_metrics_ampds(n_segments=12, dimension=24, n_streams=4, table_path='results/AMPds2/tables/'):
	total_dimension=dimension*n_streams
	streams = ['elec', 'water', 'gas', 'weather']
	distance_functions = [pairwise_euclidean, fastdtw_wrapper]
	distance_functions = [fastdtw_wrapper, pairwise_euclidean]

	for distance_function in distance_functions:
		### Data grabbing
		data = []
		for stream in range(n_streams):
			data.append([pd.read_csv(f'data/AMPds2/{streams[stream]}_segment_{segment}.csv').iloc[:,1:] for segment in range(n_segments)])
		
		combined_data = [pd.read_csv(f'data/AMPds2/all_segment_{segment}.csv').iloc[:,1:] for segment in range(n_segments)]

		### Cluster grabbing
		MSEC = pickle.load(open(f'results/AMPds2/MSEC_{distance_function.__name__}.bin','rb'))
		for stream in range(n_streams):
			for segment in range(n_segments):
				clusters = MSEC.modules[stream].get_specific_cluster_mapping(segment)
				data[stream][segment]['cluster'] = clusters

		EC = pickle.load(open(f'results/AMPds2/EC_{distance_function.__name__}.bin', 'rb'))
		for segment in range(n_segments):
			# Extract clusterings from EC object
			clustering = [-1 for x in range(len(MSEC.consensus_clusters_MV[segment]))]
			for cluster in EC.clusters[segment]:
				for index in EC.clusters[segment][cluster]:
					clustering[index] = cluster

			combined_data[segment]['cluster_MV'] = MSEC.consensus_clusters_MV[segment]
			combined_data[segment]['cluster_HGP'] = MSEC.consensus_clusters_HGP[segment]
			combined_data[segment]['cluster_MC'] = MSEC.consensus_clusters_MC[segment]
			combined_data[segment]['cluster_baseline'] = clustering

		### Distance measures
		distances = []
		combined_distances = []
		for stream in range(n_streams):
			distances.append([])
			for segment in range(n_segments):
				distances[stream].append(distance_function(data[stream][segment].iloc[:,:dimension].to_numpy(copy=True)))
		for segment in range(n_segments):
			combined_data.append(distance_function(combined_data[segment].iloc[:,:total_dimension].to_numpy(copy=True)))

		# Individual streams
		cols = columns=[f'$S^{{{x}}}$' for x in range(1,13)]
		evaluation_data = []
		for stream in range(n_streams):
			IC, SI, n_clusters = [], [], [] 
			for segment in range(n_segments):
				D = distance_function(data[stream][segment].iloc[:,:dimension].to_numpy(copy=True))
				try:
					SI.append(silhouette_score(D, data[stream][segment]['cluster'], metric='precomputed'))
				except:
					SI.append(-11)
				IC.append(IC_av(pd.DataFrame(D), data[stream][segment]['cluster'])[0])
				n_clusters.append(len(data[stream][segment]['cluster'].unique()))
			evaluation_data.append(IC)
			evaluation_data.append(SI)
			evaluation_data.append(n_clusters)

		measures = ['IC-av','SI', 'num clusters']
		indices = pd.MultiIndex.from_tuples([(f'$S_{{{stream}}}$', measure) for stream in range(n_streams) for measure in measures])
		df_scores = pd.DataFrame(evaluation_data, columns=cols, index=indices)
		s = df_scores.style
		with open(f'{table_path}{distance_function.__name__}.tex', 'w') as f:
			f.write(s.to_latex(hrules=True))

		# Consensus and baseline
		evaluation_data = []
		cluster_cols = ['cluster_MV', 'cluster_HGP', 'cluster_MC', 'cluster_baseline']
		for col in cluster_cols:
			IC, SI, n_clusters = [], [], []
			n_clusters = []
			for segment in range(n_segments):
				D = distance_function(combined_data[segment].iloc[:,:total_dimension].to_numpy(copy=True))
				try:
					SI.append(silhouette_score(D, combined_data[segment][col], metric='precomputed'))
				except:
					SI.append(-11)
				IC.append(IC_av(pd.DataFrame(D), combined_data[segment][col])[0])
				n_clusters.append(len(combined_data[segment][col].unique()))
			evaluation_data.append(SI)
			evaluation_data.append(n_clusters)

		methods = ['Majority Vote', 'Hypergraph Partitioning', 'Markov Clustering', 'Baseline']
		methods = ['MV', 'HGP', 'MC', 'Base']
		indices = pd.MultiIndex.from_tuples([(f'{method}', measure) for method in methods for measure in measures])
		df_scores = pd.DataFrame(evaluation_data, columns=cols, index=indices)
		s = df_scores.style
		with open(f'{table_path}{distance_function.__name__}_total.tex', 'w') as f:
			f.write(s.to_latex(hrules=True))


def fastdtw_wrapper(data):
    return np.array([[fastdtw(data[x], data[y])[0] for x in range(len(data))]for y in range(len(data))])

def canberra_wrapper(data):
    return pairwise_distances(data,metric='canberra')

def chebyshev_wrapper(data):
    return pairwise_distances(data,metric='chebyshev')

def correlation_wrapper(data):
    return pairwise_distances(data,metric='correlation')

def cosine_wrapper(data):
    return pairwise_distances(data,metric='cosine')

def mahalanobis_wrapper(data):
    return pairwise_distances(data,metric='mahalanobis')

def braycurtis_wrapper(data):
    return pairwise_distances(data,metric='braycurtis')

def minkowski_wrapper(data):
    return pairwise_distances(data,metric='minkowski')
def main():
	#consensus_metrics_rbf(n_segments=10, n_streams=3, dimension=2)
	#consensus_metrics_rbf(n_segments=10, n_streams=8, dimension=6)
	#consensus_metrics_rbf(n_segments=10, n_streams=12, dimension=8)
	#consensus_metrics_ampds(n_segments=12)
	consensus_metrics_ampds_new(n_segments=12)
	#consensus_metrics_ampds_subset(n_segments=12)

# Starta datagen,
# Starta clustring
# Starta denna fil. 
# Imorgon, Explorera data på tåget och plocka ut de intressanta attributerna.
# Hämta ut den intressanta datan och skapa en hyfsad clustring på den
# Fixa så att pipeline för allt fungerar
# Fixa denna fil så den kan printa ut tabeller också.
# Generalisera så olika distansfunktioner kan användas

if __name__ == '__main__':
	main()
