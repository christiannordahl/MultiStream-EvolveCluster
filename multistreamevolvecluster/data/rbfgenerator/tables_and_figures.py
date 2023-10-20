import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
import json
import matplotlib.pyplot as plt
import matplotlib.colors
import sys

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

def cluster_wise_f_measure(data):
    sum = 0
    # unique_clusters_predicted = sorted(predicted["cluster"].unique())

    # for cluster in unique_clusters_predicted:
    #     predicted_cluster = predicted[predicted["cluster"] == cluster]
    #     true_cluster = cluster_with_max_shared_experts(predicted_cluster, true)
    #     sum = sum + f_measure(set(predicted_cluster["_id"]), set(true_cluster["_id"]))
    unique_clusters_predicted = sorted(data['cluster'].unique())
    for cluster in unique_clusters_predicted:
    	predicted_cluster = data[data['cluster'] == cluster]
    	true_cluster = cluster_with_max_shared(predicted_cluster, data)
    	sum += f_measure(set(predicted_cluster['_id']), set(true_cluster['_id']))

    length = len(unique_clusters_predicted)
    return (sum / length)

def f_measure(pred, true):
    value = (2 * len(pred & true)) / (len(true) + len(pred))
    return value

def cluster_with_max_shared(predicted_cluster, data):
	predicted = set(predicted_cluster['_id'])
	true = data['actual_cluster'].unique()
	max_cluster = 0
	max_session = 0

	for label in true:
		cluster = data[data['actual_cluster'] == label]
		same_cluster = len(predicted & set(cluster['_id']))
		if same_cluster > max_cluster:
			max_cluster = same_cluster
			max_session = label
	return data[data['actual_cluster'] == max_session]
	
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


def main():
	seeds = [1,2,6,29]
	seeds = [x for x in range(100)]
	i = 0
	n_streams=3
	n_segments=10
	createdelete=False
	figures_and_tables_single(seeds, n_streams, n_segments, createdelete)
	#createdelete=True
	#figures_and_tables_single(seeds, n_streams, n_segments, createdelete)

if __name__ == '__main__':
	main()
