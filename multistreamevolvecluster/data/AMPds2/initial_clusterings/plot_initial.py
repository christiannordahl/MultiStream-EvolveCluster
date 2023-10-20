import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hours = [1,2,3,4,6,8]
streams = ['elec', 'gas', 'water', 'weather', 'all']
dist_meas = ['canberra', 'minkowski', 'pairwise_euclidean', 'fastdtw_wrapper']

for hour in hours:
	for stream in streams:
		df_total = pd.DataFrame()
		for dist in dist_meas:
			df = pd.read_csv(f'{hour}H_initial_{stream}_{dist}_seg_0.csv', index_col=0)
			df_total[f'{dist}'] = df['sil']

		plot = df_total.plot(ylim=(0,1), alpha=0.7, title=f'{stream}_{hour}H')
		plot = plot.get_figure()
		plot.savefig(f'plots/{stream}_{hour}H.pdf')
		plt.close()


