import pandas as pd
from scipy.stats import zscore

columns = ['Pt', 'avg_rate', 'avg_rate', 'Temp (C)']
files = ['Electricity_WHE.csv', 'Water_WHW.csv', 'NaturalGas_WHG.csv', 'Climate_HourlyWeather.csv']
filepaths = [f'original/{x}' for x in files]

hours = [1,2,3,4,6,8]

for hour in hours:
	dfs=[]
	zdfs=[]
	i=0
	for file, cols in zip(filepaths, columns):
		df = pd.read_csv(file)
		if cols in ['Pt', 'avg_rate']:
			df['unix_ts'] = pd.to_datetime(df['unix_ts'], unit='s')
			df.rename(columns={'unix_ts':'timestamp', cols:'value'}, inplace=True)
		else:
			df['Date/Time'] = pd.to_datetime(df['Date/Time'])
			df.rename(columns={'Date/Time':'timestamp', cols:'value'}, inplace=True)
		df.set_index('timestamp', inplace=True)
		
		df = df[['value']]
		if cols == 'Pt':
			# Remakes the data so that it is consumption since last time stamp and converts to kWh
			df['value'] = df['value'].diff().fillna(0).divide(1000)
			df = df.resample(f'{hour}H').sum()
		elif cols == 'avg_rate':
			df = df.resample(f'{hour}H').sum()
			if i == 2:	# For the Gas we also convert by division of 1000
				df = df.divide(1000)
		elif cols=='Temp (C)':
			df = df.resample(f'{hour}H').mean()
		
		start = df.index[0]
		start = start.replace(hour=0, minute=0)
		end = df.index[-1]
		end = end.replace(hour=0, minute=0)

		df.drop(df[(df.index.day == start.day) & (df.index.month == start.month) & (df.index.year == start.year)].index, inplace=True)
		df.drop(df[(df.index.day == end.day) & (df.index.month == end.month) & (df.index.year == end.year)].index, inplace=True)
		idx = pd.date_range(df.index[0], df.index[-1], freq='D')
		
		dfs.append(df)
		i+=1

	# Filter out any timestamp that isnt available in all streams.
	# There are 22 nan values in the weather data. Linear interpolation to fix it.
	start = dfs[0].index[0]
	end = dfs[0].index[-1]
	for df in dfs:
		if df.index[0] > start:
			start = df.index[0]
		if df.index[-1] < end:
			end = df.index[-1]
	for i in range(len(dfs)):
		dfs[i] = dfs[i].loc[start:end]
		if dfs[i].isnull().values.any():
			dfs[i]['value'] = dfs[i]['value'].interpolate('linear')
		if dfs[i].isnull().values.any():
			print('is still null...')
		# Normalizing
		dfs[i]['value'] = (dfs[i]['value'] - dfs[i]['value'].min())/(dfs[i]['value'].max() - dfs[i]['value'].min())


	# Reshaping into profiles.
	files = ['elec.csv', 'water.csv', 'gas.csv', 'weather.csv']
	abbreviations = ['E','Wa','G','We']
	for i in range(len(dfs)):
		print(len(dfs[i]))
		dfs[i] = pd.DataFrame(dfs[i].values.reshape(-1, (24//hour)), index=idx)
		dfs[i].to_csv(f'{files[i]}')
		dfs[i].rename(columns={x:f'{abbreviations[i]}{x}' for x in range(24//hour)}, inplace=True)

	total_df = dfs[0]
	for i in range(1,4):
		total_df = total_df.join(dfs[i])
	total_df.to_csv('all.csv')

	dates = ['2012-04-01', '2012-06-01', '2012-08-01', '2012-10-01', '2012-12-01', 
			 '2013-02-01', '2013-04-01', '2013-06-01', '2013-08-01', '2013-10-01', '2013-12-01', 
			 '2014-02-01', '2014-04-01']

	for i in range(0, len(dates)-1):
		total_df.loc[dates[i]:dates[i+1],:].to_csv(f'{hour}H_all_segment_{i}.csv')
		total_df.loc[dates[i]:dates[i+1],:].apply(zscore, axis=0).to_csv(f'z_{hour}H_all_segment_{i}.csv')
		for df, file in zip(dfs, files):
			df.loc[dates[i]:dates[i+1],:].to_csv(f'{hour}H_{file.split(".")[0]}_segment_{i}.csv')
			df.loc[dates[i]:dates[i+1],:].apply(zscore, axis=0).to_csv(f'z_{hour}H_{file.split(".")[0]}_segment_{i}.csv')
