from EvolveCluster import EvolveCluster
from functions import pairwise_euclidean, pairwise_hamming, pairwise_distances
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import json
import kahypar as kahypar
import pickle
from time import perf_counter
import matplotlib.pyplot as plt

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import subprocess

class MultiStreamEvolveCluster(object):
    """docstring for MultiStreamEvolveCluster"""
    def __init__(self, data, clusters, centroids, tau, num_modules, distance_functions, distances=None, combined_influence=False, threshold=0.5):
        '''
        1. Should the initial clusters and centroids be based from a total clustering solution
            or smaller solutions? Make it so that it works for both.
        2. We should allow to set distance functions to each individual object, i.e. one could 
            use ED, one DTW, one manhattan etc. This allows for better division of features.
        3. Data should be sent in already divided. I.e. a list of lists of data or list of dataframes.
        4. Error check for each list being of the num_modules length.
        5. Distances cannot be pre-computed. Let each EC object calculate and stores its distances.
        6. 
        '''
        if type(num_modules) is not int:
            raise TypeError('Please enter num_modules as an int.')
        elif type(data) is not list or len(data) != num_modules:
            raise TypeError('Please make sure data is a list and has the length num_modules. num_modules = %d'%(num_modules))
        elif type(clusters) is not list or len(clusters) != num_modules:
            raise TypeError('Please make sure clusters is a list and has the length num_modules. num_modules =%d'%(num_modules))
        elif type(centroids) is not list or len(centroids) != num_modules:
            raise TypeError('Please make sure centroids is a list and has the length num_modules. num_modules =%d'%(num_modules))
        elif type(tau) is not list or len(tau) != num_modules:
            raise TypeError('Please make sure tau is a list and has the length num_modules. num_modules =%d'%(num_modules))

        self.threshold = threshold
        self.num_modules = num_modules
        self.combined_influence = combined_influence
        self.modules = [EvolveCluster(data[i], clusters[i], centroids[i], tau[i], distances=None, distance_function=distance_functions[i]) for i in range(self.num_modules)]
        self.consensus_clusters = []
        self.consensus_clusters_MV = []
        self.consensus_clusters_HGP = []
        self.consensus_clusters_MC = []

    def cluster_entire_stream(self, data_segments):
        '''
        data_segments should a list of lists of numpy arrays.
        The outer list represents the entire stream.
        The inner lists (i.e. data_segments[i]) represents a single segment in the stream.
        Each numpy array is then an individual view of a single segment.
        '''
        for data_segment in data_segments:
            self.cluster_stream_segment(data_segment)

    def cluster_entire_stream_only(self, data_streams):
        i = 0
        for data_stream, module in zip(data_streams, self.modules):
            print(f'stream {i}')
            module.cluster_stream(data_stream)
            i += 1
            
    def cluster_stream_segment_only(self, data_segment):
        '''
        data_segment should a list of numpy arrays.
        The list represents a single segment in the stream.
        Each numpy array is then an individual view of a single segment.
        '''

        # This loop could be paralellized to both simulate a truly distributed algorithm,
        # but also to improve performance, as now it is sequentially clustered.
        i = 0
        for segment, module in zip(data_segment, self.modules):
            module.cluster_segment(segment)
            i+=1

    def consensus_post(self, n_segments, post=False):
        self.majority_vote_post(n_segments)
        self.MCLA_post(n_segments)
        self.HGPA_post(n_segments)

    def consensus_investigation(self, n_streams, n_segments, I):
        self.MCLA_investigation(n_streams, n_segments, I)
        self.HGPA_investigation(n_streams, n_segments)

    def cluster_stream_segment(self, data_segment):
        '''
        data_segment should a list of numpy arrays.
        The list represents a single segment in the stream.
        Each numpy array is then an individual view of a single segment.
        '''

        # This loop could be paralellized to both simulate a truly distributed algorithm,
        # but also to improve performance, as now it is sequentially clustered.
        i = 0
        for segment, module in zip(data_segment, self.modules):
            module.cluster_segment(segment)
            i+=1

        self.majority_vote()    # Simplest form of combining an ensemble of solutions to a global one
        self.HGPA()            # Hypergraph partitioning to form a global clustering solution by cutting hyperedges
        self.MCLA()            # Represents clusters by hyperedges and collapses hyperedges to crate k clusters

        # If we intend to use the combined clustering solutions centroids to pass information
        # to each module instead of allowing each module to self operate with its own centroids
        # from previous segments.
        if self.combined_influence is True:
            self.influence_modules()

    def influence_modules(self):
        # Passes cluster centroids from the combined clustering solution
        # to each module instead of its previous centroids.
        # Not implemented. Just a train of thought for specific applications in future works.
        pass

    def HGPA_post(self, n_segments):
        self.consensus_clusters_HGP = []
        for segment in range(n_segments):
            clusterings = [self.modules[i].get_specific_cluster_mapping(segment) for i in range(self.num_modules)]
            num_nodes = len(clusterings[0])
            num_nets = len(clusterings)

            # You should calculate the hyperedges from clusterings above
            # Below is the easy way instead
            hyperedges = []
            hyperedge_indices = [0]
            k = 0
            num_nets = 0
            for module in self.modules:
                if len(module.clusters[segment]) > k:
                    k = len(module.clusters[segment])
                for cluster in module.clusters[segment]:
                    hyperedges += list(module.clusters[segment][cluster])
                    hyperedge_indices.append(len(hyperedges))
                    num_nets+=1

            node_weights = [1 for x in range(num_nodes)]
            edge_weights = [1 for x in range(num_nets)]

            hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, k, edge_weights, node_weights)
            # Do we need to set a random state here aswell, for reproducibility?
            context = kahypar.Context()
            context.loadINIconfiguration("km1_kKaHyPar_sea20.ini")
            context.setK(k)
            context.setEpsilon(0.03)
            context.suppressOutput(True)    # Supresses the output.

            print(num_nets)
            for module in self.modules:
                print('.',len(module.clusters[segment]))

            kahypar.partition(hypergraph, context)
            

            clusters = [hypergraph.blockID(x) for x in range(num_nodes)]
            self.consensus_clusters_HGP.append(clusters)

    def HGPA_investigation(self, n_streams, n_segments):
        self.consensus_clusters_HGP = []
        for segment in range(n_segments):
            clusterings = [self.modules[i].get_specific_cluster_mapping(segment) for i in range(self.num_modules)]
            num_nodes = len(clusterings[0])
            num_nets = len(clusterings)

            # You should calculate the hyperedges from clusterings above
            # Below is the easy way instead
            hyperedges = []
            hyperedge_indices = [0]
            k = 0
            num_nets = 0
            for module in self.modules[:n_streams]:
                if len(module.clusters[segment]) > k:
                    k = len(module.clusters[segment])
                for cluster in module.clusters[segment]:
                    hyperedges += list(module.clusters[segment][cluster])
                    hyperedge_indices.append(len(hyperedges))
                    num_nets+=1

            node_weights = [1 for x in range(num_nodes)]
            edge_weights = [1 for x in range(num_nets)]

            hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, k, edge_weights, node_weights)
            # Do we need to set a random state here aswell, for reproducibility?
            context = kahypar.Context()
            context.loadINIconfiguration("km1_kKaHyPar_sea20.ini")
            context.setK(k)
            context.setEpsilon(0.03)
            context.suppressOutput(True)    # Supresses the output.

            kahypar.partition(hypergraph, context)
            

            clusters = [hypergraph.blockID(x) for x in range(num_nodes)]
            self.consensus_clusters_HGP.append(clusters)
    
    def MCLA_post(self, n_segments):
        self.consensus_clusters_MC = []

        # Create format and save to file.
        # Rub subprocess
        # Extract the data from the file into the list here.
        r_val = 1/(self.num_modules+1)  # Exponential
                                        # Överväga threshold. SÅ om ex. mindre än 2 strömmar har sagt att de har en båge så slopar vi den.


        for segment in range(n_segments):
            length = sum([len(self.modules[0].clusters[segment][x]) for x in self.modules[0].clusters[segment]])
            co_assoc = np.ones((length,length))
            #co_assoc = np.zeros((len(self.modules[0].data[-1]),len(self.modules[0].data[-1])))
            for module in self.modules:
                print(len(module.clusters), segment)
                for cluster in module.clusters[segment]:
                    cols = [x for x in module.clusters[segment][cluster]]
                    rows = [[x] for x in cols]
                    co_assoc[rows,cols] -= r_val

            data = []
            for row in range(length):
                data.append([])
                for col in range(length):
                    if co_assoc[row,col] < 1:
                        data[-1].append((col,co_assoc[row,col]))
            with open('mcl_data.txt', 'w') as f:
                for row in range(length):
                    for item in data[row]:
                        f.write(f'{row} {item[0]} {item[1]}\n') 
                        #                 f.write(f'''(mclheader
                        # mcltype matrix
                        # dimensions 1000x1000
                        # )
                        # (mcldoms
                        # ''')
                        #                 for i in range(length):
                        #                     f.write(f'{i} ')
                        #                 f.write('''$
                        # )
                        # (mclmatrix
                        # begin
                        # ''')
                        #                 for row in range(length):
                        #                     f.write(f'{row}\t')
                        #                     for item in data[row]:
                        #                         f.write(f'{item[0]}:{item[1]}\t')
                        #                     f.write('$+\n')
                        #                 f.write('''
                        # )''')



            p = subprocess.Popen("./mcl mcl_data.txt -I 5.0 --abc -o mcl_output.txt", shell=True)
            p.wait()

            clusters = [-1 for x in range(length)]
            with open('mcl_output.txt', 'r') as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    items = [int(item.rstrip().lstrip()) for item in lines[i].split()]
                    for item in items:
                        clusters[item] = i
            self.consensus_clusters_MC.append(clusters)

    def MCLA_investigation(self, n_streams, n_segments, I):
        self.consensus_clusters_MC = []

        # Create format and save to file.
        # Rub subprocess
        # Extract the data from the file into the list here.
        r_val = 1/(n_streams+1)
        for segment in range(n_segments):
            length = sum([len(self.modules[0].clusters[segment][x]) for x in self.modules[0].clusters[segment]])
            #co_assoc = np.ones((length,length))
            co_assoc = np.full((length, length), float(2**n_streams))

            for module in self.modules[:n_streams]:
                print(len(module.clusters), segment)
                for cluster in module.clusters[segment]:
                    cols = [x for x in module.clusters[segment][cluster]]
                    rows = [[x] for x in cols]
                    co_assoc[rows,cols] /= 2

            data = []
            for row in range(length):
                data.append([])
                for col in range(length):
                    if co_assoc[row,col] < (2**(n_streams)):
                        data[-1].append((col,co_assoc[row,col]))
            with open('mcl_data.txt', 'w') as f:
                for row in range(length):
                    for item in data[row]:
                        f.write(f'{row} {item[0]} {item[1]}\n') 
                        
            p = subprocess.Popen(f"./mcl mcl_data.txt -I {I} --abc -o mcl_output.txt", shell=True)
            p.wait()

            clusters = [-1 for x in range(length)]
            with open('mcl_output.txt', 'r') as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    items = [int(item.rstrip().lstrip()) for item in lines[i].split()]
                    for item in items:
                        clusters[item] = i
            self.consensus_clusters_MC.append(clusters)

    def majority_vote_post(self, n_segments):
        self.consensus_clusters_MV = [] # Resets the list

        r_val = 1/self.num_modules
        for segment in range(n_segments):
            length = sum([len(self.modules[0].clusters[segment][x]) for x in self.modules[0].clusters[segment]])
            co_assoc = np.zeros((length,length))
            #co_assoc = np.zeros((len(self.modules[0].data[-1]),len(self.modules[0].data[-1])))
            for module in self.modules:
                for cluster in module.clusters[segment]:
                    cols = [x for x in module.clusters[segment][cluster]]
                    rows = [[x] for x in cols]
                    co_assoc[rows,cols] += r_val
            np.fill_diagonal(co_assoc, 0)       # Removes diagonal, I don't think we have to but why not.
            singletons = np.where((co_assoc < self.threshold).all(axis=-1))   # If there is an item that does not have a consistent pair with another item
            pairs = np.transpose((co_assoc >= self.threshold).nonzero())     # Identifies all pairs (association score higher than threshold)

            # Adds all paired items to the corresponding cluster. If items that have previously belonged to separate clusters are paired together, then the 
            # both entire clusters are merged.
            clusters = []
            for pair in pairs:
                c1 = None
                c2 = None
                for i in range(len(clusters)):
                    if pair[0] in clusters[i]:
                        c1 = i
                    if pair[1] in clusters[i]:
                        c2 = i
                if c1 is None and c2 is None:
                    clusters.append({})
                    clusters[-1][pair[0]] = None
                    clusters[-1][pair[1]] = None
                elif c1 is None and c2 is not None:
                    clusters[c2][pair[0]] = None
                elif c1 is not None and c2 is None:
                    clusters[c1][pair[1]] = None
                elif c1 is not None and c2 is not None and c1 != c2:
                    clusters[c1] = clusters[c1] | clusters[c2]

            # Converts from quick access format to the format we have in EvolveCluster (i.e. a dict where cluster number is the key and value is list of elements
            # belonging to said cluster)
            C = {}
            for i in range(len(clusters)):
                C[i] = []
                for value in clusters[i].keys():
                    C[i].append(value)
            # Creates singleton clusters for remaining items that have no pairs.
            for singleton in singletons:
                for s in singleton:
                    C[i] = [s]
                    i += 1

            clusters = [-1 for x in range(length)]
            for cluster in C:
                for index in C[cluster]:
                    clusters[index] = cluster

            self.consensus_clusters_MV.append(clusters)

    def majority_vote_post_less_singletons(self):
        # Reassign clusters to follow module[0]
        # H = []
        # num_elements = len(self.modules[0].data[-1])
        # for module in self.modules:
        #     H.append(np.zeros((len(module.clusters[-1].keys()), num_elements), dtype=int))
        #     for key in module.clusters[-1].keys():
        #         H[-1][int(key)][np.array(module.clusters[-1][key])] = 1
        # h = H[0].copy()
        # for arr in H[1:]:
        #     h = np.append(h,arr,axis=0)
        # h = np.transpose(h)
        # H = None
        r_val = 1/self.num_modules
        for segment in range(self.modules[0].nr_increments):
            length = sum([len(self.modules[0].clusters[segment][x]) for x in self.modules[0].clusters[segment]])
            co_assoc = np.zeros((length,length))
            #co_assoc = np.zeros((len(self.modules[0].data[-1]),len(self.modules[0].data[-1])))
            for module in self.modules:
                print(len(module.clusters), segment)
                for cluster in module.clusters[segment]:
                    cols = [x for x in module.clusters[segment][cluster]]
                    rows = [[x] for x in cols]
                    co_assoc[rows,cols] += r_val
            np.fill_diagonal(co_assoc, 0)       # Removes diagonal, I don't think we have to but why not.
            singletons = np.where((co_assoc > self.threshold).all(axis=-1))   # If there is an item that does not have a consistent pair with another item
            pairs = np.transpose((co_assoc >= self.threshold).nonzero())     # Identifies all pairs (association score higher than threshold)
            
            # Lowers threshold, hoping to identify some cluster to put singletons
            new_co_assoc = np.delete(co_assoc, np.where(co_assoc >= self.threshold)[0], axis=0)
            new_co_assoc = np.delete(new_co_assoc, np.where(co_assoc >= self.threshold)[1], axis=1)

            print(new_co_assoc)         # This needs a little bit of continued work. I need some data that produces singletons.
            if len(new_co_assoc) > 0:   # But, should be just to find all lower pairs that have a value > 0, or 0.3, or whatever. 
                print(new_co_assoc)     # These lower pairs should be added to the corresponding clusters. The remainder after that should be singletons.
                lower_pairs = np.argmax(new_co_assoc, axis=0)
                print(lower_pairs)
            return
            # Adds all paired items to the corresponding cluster. If items that have previously belonged to separate clusters are paired together, then the 
            # both entire clusters are merged.
            clusters = []
            for pair in pairs:
                c1 = None
                c2 = None
                for i in range(len(clusters)):
                    if pair[0] in clusters[i]:
                        c1 = i
                    if pair[1] in clusters[i]:
                        c2 = i
                if c1 is None and c2 is None:
                    clusters.append({})
                    clusters[-1][pair[0]] = None
                    clusters[-1][pair[1]] = None
                elif c1 is None and c2 is not None:
                    clusters[c2][pair[0]] = None
                elif c1 is not None and c2 is None:
                    clusters[c1][pair[1]] = None
                elif c1 is not None and c2 is not None and c1 != c2:
                    clusters[c1] = clusters[c1] | clusters[c2]

            # Converts from quick access format to the format we have in EvolveCluster (i.e. a dict where cluster number is the key and value is list of elements
            # belonging to said cluster)
            C = {}
            for i in range(len(clusters)):
                C[i] = []
                for value in clusters[i].keys():
                    C[i].append(value)
            # Creates singleton clusters for remaining items that have no pairs.
            for singleton in singletons:
                for s in singleton:
                    C[i] = [s]
                    i += 1

            self.consensus_clusters_MV.append(C)

def dev_test():
    dfs = [[pd.read_csv('data/s1/continuous/%d.csv'%(i)) for j in range(2)]for i in range(5)]

    distance_functions = [pairwise_euclidean for i in range(2)]
    for dfss in dfs:
        for df in dfss:
            df['cluster'] -= 1
    with open('data/s1/continuous/%d.txt'%(0), 'r') as f:
        a = f.readlines()
        f.close()
    a = [int(x) for x in a[2].rstrip().split(',')]
    medoids = sorted(list(set(a)))
    C = {}
    for i in range(len(medoids)):
        C[str(i)] = []
        for j in range(len(a)):
            if a[j] == medoids[i]:
                C[str(i)].append(j)
        C[str(i)] = np.array(C[str(i)])

    C = [C.copy() for x in range(2)]
    medoids = [medoids.copy() for x in range(2)]

    MSEC = MultiStreamEvolveCluster(data=[dfs[0][0].iloc[:,:2].to_numpy(copy=True), dfs[1][0].iloc[:,:2].to_numpy(copy=True)], 
                            clusters=C, centroids=medoids, tau=[0.08, 0.08], num_modules=2, distance_functions=distance_functions)
    
    dfss = dfs[1:]
    for d in dfss:
        for i in range(len(d)):
            d[i] = d[i].iloc[:,:2].to_numpy(copy=True)
    MSEC.cluster_entire_stream(dfss)

def dev_test_separate_streams():
    for j in range(2):
        with open('data/s1/continuous/%d.csv'%(i)) as f:
            dfs = [pd.read_csv('data/s1/continuous/%d.csv'%(i)) for i in range(5)]

        distance_functions = [pairwise_euclidean for i in range(2)]
        for dfss in dfs:
            for df in dfss:
                df['cluster'] -= 1
        with open('data/s1/continuous/%d.txt'%(0), 'r') as f:
            a = f.readlines()
            f.close()
        a = [int(x) for x in a[2].rstrip().split(',')]
        medoids = sorted(list(set(a)))
        C = {}
        for i in range(len(medoids)):
            C[str(i)] = []
            for j in range(len(a)):
                if a[j] == medoids[i]:
                    C[str(i)].append(j)
            C[str(i)] = np.array(C[str(i)])

        C = [C.copy() for x in range(2)]
        medoids = [medoids.copy() for x in range(2)]

        MSEC = MultiStreamEvolveCluster(data=[dfs[0][0].iloc[:,:2].to_numpy(copy=True), dfs[1][0].iloc[:,:2].to_numpy(copy=True)], 
                                clusters=C, centroids=medoids, tau=[0.08, 0.08], num_modules=2, distance_functions=distance_functions)
    
    dfss = dfs[1:]
    for d in dfss:
        for i in range(len(d)):
            d[i] = d[i].iloc[:,:2].to_numpy(copy=True)
    MSEC.cluster_entire_stream(dfss)

def rbfgenerator_tests(dimension=2, n_streams=3):
    seeds = [75]
    n_segments=10
    tau = [0.1 for x in range(n_streams)]
    for seed in seeds:
        dfs = [[pd.read_csv(f'data/rbfgenerator/{dimension}-dim/seed_{seed}_stream_{stream}_segment_{segment}_createdelete=False.csv') for segment in range(n_segments)]for stream in range(n_streams)]
        for stream in range(n_streams):
            dfs[stream][0]['true_cluster'] = dfs[stream][0]['cluster']
            for segment in range(1,n_segments):
                dfs[stream][segment].rename({'cluster':'true_cluster'})

        distance_functions = [pairwise_euclidean for i in range(n_streams)]
        C = [{} for x in range(n_streams)]
        medoids = [[] for x in range(n_streams)]

        for stream in range(n_streams):
            tmp = dfs[stream][0].iloc[:,:dimension].to_numpy(copy=True)
            D = pairwise_euclidean(tmp)
            df = dfs[0][0]
            df['clusters'] = np.nan
            for i in range(5):
                test = df[df['cluster'] == i]
                D_test = pairwise_euclidean(test.iloc[:,:dimension].to_numpy(copy=True))
                medoid = np.argmin(D_test.sum(axis=0))
                medoid = int(test.iloc[medoid].name)
                df.loc[df['cluster'] == i, 'clusters'] = medoid
                medoids[stream].append(medoid)

            for cluster in range(len(medoids[stream])):
                C[stream][str(cluster)] = np.array(sorted(df[df['cluster'] == cluster].index.tolist()))
        
        dfss=[]
        for i in range(len(dfs)):
            dfss.append(dfs[i][1:])
        for d in dfss:
            for i in range(len(d)):
                d[i] = d[i].iloc[:,:dimension].to_numpy(copy=True)
        
        data = [dfs[stream][0].iloc[:,:dimension].to_numpy(copy=True) for x in range(n_streams)]
        time_before_1 = perf_counter()
        MSEC = MultiStreamEvolveCluster(data=data,
                                      clusters=C,
                                      centroids=medoids,
                                      tau=tau,
                                      num_modules=n_streams,
                                      distance_functions=distance_functions)
        time_after_1 = perf_counter()

        print(f'seed: {seed}')
        time_before_2 = perf_counter()
        MSEC.cluster_entire_stream_only(dfss)
        time_after_2 = perf_counter()
        pickle.dump(MSEC, open(f'results/rbfgenerator/{dimension}-dim/MSEC_seed_{seed}.bin','bw'))
        print('time creating MSEC object', time_after_1-time_before_1)
        print('time clustering streams', time_after_2-time_before_2)

    # Baseline
    total_dimension = dimension*n_streams

    for seed in seeds:
        dfs = [[pd.read_csv(f'data/rbfgenerator/{dimension}-dim/seed_{seed}_stream_{stream}_segment_{segment}_createdelete=False.csv') for segment in range(n_segments)]for stream in range(n_streams)]
        new_dfs = []
        for segment in range(n_segments):
            new_dfs.append(dfs[0][segment])
            for stream in range(1, n_streams):
                for i in range(dimension):
                    new_dfs[segment][f'{i+1+stream*dimension}'] = dfs[stream][segment][f'{i}']
            cols = new_dfs[segment].columns.tolist()
            cols = cols[:dimension]+cols[(dimension+1):]+[cols[dimension]]
            new_dfs[segment] = new_dfs[segment][cols]
        dfs = new_dfs

        # Here we create an EvolveCluster object and run it on the 6 dim data.
        # we should then create another function where we misalign the cluster labels.
        C = {} 
        medoids = []
        tmp = dfs[0].iloc[:,:total_dimension].to_numpy(copy=True)
        D = pairwise_euclidean(tmp)
        df = dfs[0]
        df['clusters'] = -1

        for i in range(5):
            test = df[df['cluster'] == i]
            D_test = pairwise_euclidean(test.iloc[:,:total_dimension].to_numpy(copy=True))
            medoid = np.argmin(D_test.sum(axis=0))
            medoid = int(test.iloc[medoid].name)
            df.loc[df['cluster'] == i, 'clusters'] = medoid
            medoids.append(medoid)
        df['clusters'] = df['clusters'].astype('int')

        for cluster in range(len(medoids)):
            C[str(cluster)] = np.array(sorted(df[df['cluster'] == cluster].index.tolist()))

        dfss = []
        for d in dfs[1:]:
            dfss.append(d.iloc[:,:total_dimension].to_numpy(copy=True))
        data = dfs[0].iloc[:,:total_dimension].to_numpy(copy=True)

        time_before_1 = perf_counter()
        ec = EvolveCluster(data=data, 
                           clusters=C,
                           centroids=medoids,
                           tau=0.1,
                           distances=None,
                           distance_function=pairwise_euclidean)
        time_after_1 = perf_counter()
        print(f'seed: {seed}')
        time_before_2 = perf_counter()
        ec.cluster_stream(dfss)
        time_after_2 = perf_counter()

        pickle.dump(ec, open(f'results/rbfgenerator/{dimension}-dim/EC_seed_{seed}.bin', 'bw'))
        print('time creating EC object', time_after_1-time_before_1)
        print('time clustering stream', time_after_2-time_before_2)

def rbf_generator_part2(dimension=2,n_streams=3, n_segments=10):
    # HEre we load everything in and perform the consensus on each step.
    seeds = [75]
    for seed in seeds:
        MSEC = pickle.load(open(f'results/rbfgenerator/{dimension}-dim/MSEC_seed_{seed}.bin','rb'))
        MSEC.consensus_post(n_segments)
        pickle.dump(MSEC, open(f'results/rbfgenerator/{dimension}-dim/MSEC_seed_{seed}.bin','wb'))

def extract_clusterings_rbf(dimension=2, n_streams=3):
    seeds = [75]
    n_segments=10
    for seed in seeds:
        MSEC = pickle.load(open(f'results/rbfgenerator/{dimension}-dim/MSEC_seed_{seed}.bin','rb'))
        dfs = [[pd.read_csv(f'data/rbfgenerator/{dimension}-dim/seed_{seed}_stream_{stream}_segment_{segment}_createdelete=False.csv') for segment in range(n_segments)]for stream in range(n_streams)]
        for stream in range(n_streams):
            for segment in range(1,n_segments):
                print(MSEC.modules[stream].get_specific_cluster_mapping(segment))

def ampds_tests_part1():
    best_initial = {'elec':[4,4,'pairwise_euclidean', pairwise_euclidean],
                    'water':[8,3,'pairwise_euclidean',pairwise_euclidean],
                    'gas':[6,3,'pairwise_euclidean',pairwise_euclidean],
                    'weather':[4,4,'canberra',canberra_wrapper]}

    n_segments = 12
    n_streams = 4
    streams = ['elec', 'water', 'gas', 'weather']
    dimensions = [24//best_initial[stream][0] for stream in streams]
    tau = [0.3, 0.37, 0.07, 0.1]
    #tau = [0.32, 0.15, 0.08, 0.1]
    
    dist_funcs = [pairwise_euclidean, fastdtw_wrapper]
    distance_functions = ['canberra', 'chebyshev', 'mahalanobis', 'minkowski']


    dfs = [[pd.read_csv(f'data/AMPds2/{best_initial[stream][0]}H_{stream}_segment_{segment}.csv').iloc[:,1:] for segment in range(n_segments)] for stream in streams]

    medoids = []
    for stream in range(n_streams):
        if isinstance(best_initial[streams[stream]][2], str):
            initial_clusterings = [int(x) for x in pd.read_csv(f'data/AMPds2/initial_clusterings/{best_initial[streams[stream]][0]}H_initial_{streams[stream]}_{best_initial[streams[stream]][2]}_seg_0.csv').loc[best_initial[streams[stream]][1]]['sil_clusters'][1:-1].split(',')]
        else:
            initial_clusterings = [int(x) for x in pd.read_csv(f'data/AMPds2/initial_clusterings/{best_initial[streams[stream]][0]}H_initial_{streams[stream]}_{best_initial[streams[stream]][2].__name__}_seg_0.csv').loc[best_initial[streams[stream]][1]]['sil_clusters'][1:-1].split(',')]

        medoids.append(sorted(list(set(initial_clusterings))))
        for x in range(len(medoids[-1])):
            for ic in range(len(initial_clusterings)):
                if initial_clusterings[ic] == medoids[-1][x]:
                    initial_clusterings[ic] = x
            
        dfs[stream][0]['cluster'] = initial_clusterings

    distance_functions = [best_initial[stream][3] if len(best_initial[stream]) == 4 else best_initial[stream][2] for stream in streams]
    C = [{} for x in range(n_streams)]

    for stream in range(n_streams):
        for cluster in range(len(medoids[stream])):
            C[stream][str(cluster)] = np.array(sorted(dfs[stream][0][dfs[stream][0]['cluster'] == cluster].index.tolist()))
    
    dfss=[]
    for i in range(len(dfs)):
        dfss.append(dfs[i][1:])
    j = 0
    for d in dfss:
        for i in range(len(d)):
            d[i] = d[i].iloc[:,:dimensions[j]].to_numpy(copy=True)
        j += 1
    
    data = [dfs[stream][0].iloc[:,:dimensions[stream]].to_numpy(copy=True) for stream in range(n_streams)]

    time_before_1 = perf_counter()
    MSEC = MultiStreamEvolveCluster(data=data,
                                  clusters=C,
                                  centroids=medoids,
                                  tau=tau,
                                  num_modules=n_streams,
                                  distance_functions=distance_functions)
    time_after_1 = perf_counter()
    time_before_2 = perf_counter()

    MSEC.cluster_entire_stream_only(dfss)
    time_after_2 = perf_counter()
    pickle.dump(MSEC, open(f'results/AMPds2/MSEC.bin','bw'))
    print('time creating MSEC object', time_after_1-time_before_1)
    print('time clustering streams', time_after_2-time_before_2)

def ampds_tests_baseline():
    n_segments = 12
    n_streams = 4
    dimension = 24
    streams = ['elec', 'water', 'gas', 'weather']
    dfs = [[pd.read_csv(f'data/AMPds2/{stream}_segment_{segment}.csv').iloc[:,1:] for segment in range(n_segments)] for stream in streams]

    #hours = [1,2,4,6,8]
    hours = [4]
    #num_clusts = [3,6,5,4,5]
    num_clusts = [5]
    #dist_funcs = [minkowski_wrapper, pairwise_euclidean, minkowski_wrapper, minkowski_wrapper, pairwise_euclidean]
    dist_funcs = [minkowski_wrapper]
    #func_names = ['minkowski','pairwise_euclidean','minkowski','minkowski','pairwise_euclidean']
    func_names = ['minkowski']

    for hour, num_clust, distance_function, func_name in zip(hours, num_clusts, dist_funcs, func_names):
        total_dimension = 24//hour*4
        dfs = [pd.read_csv(f'data/AMPds2/{hour}H_all_segment_{segment}.csv').iloc[:,1:] for segment in range(n_segments)]
        C = {} 
        medoids = []
        initial_clusterings = [int(x) for x in pd.read_csv(f'data/AMPds2/initial_clusterings/{hour}H_initial_all_{func_name}_seg_0.csv').loc[num_clust]['sil_clusters'][1:-1].split(',')]
        medoids = sorted(list(set(initial_clusterings)))
        tau = 0.07

        for x in range(len(medoids)):
            for ic in range(len(initial_clusterings)):
                if initial_clusterings[ic] == medoids[x]:
                    initial_clusterings[ic] = x
            # Byt ut medoid i initial_clustering till medoid indexet, så vi har 0-x.
        dfs[0]['cluster'] = initial_clusterings

        for cluster in range(len(medoids)):
            C[str(cluster)] = np.array(sorted(dfs[0][dfs[0]['cluster'] == cluster].index.tolist()))

        dfss = []
        for d in dfs[1:]:
            dfss.append(d.iloc[:,:total_dimension].to_numpy(copy=True))
        data = dfs[0].iloc[:,:total_dimension].to_numpy(copy=True)

        time_before_1 = perf_counter()
        ec = EvolveCluster(data=data, 
                           clusters=C,
                           centroids=medoids,
                           tau=tau,
                           distances=None,
                           distance_function=distance_function)
        time_after_1 = perf_counter()
        time_before_2 = perf_counter()

        ec.cluster_stream(dfss)
        time_after_2 = perf_counter()

        pickle.dump(ec, open(f'results/AMPds2/EC.bin', 'bw'))
        print('time creating EC object', time_after_1-time_before_1)
        print('time clustering stream', time_after_2-time_before_2)

def ampds_part2_removing_streams(n_segments=12):
    # HEre we load everything in and perform the consensus on each step.
    MSEC = pickle.load(open('results/AMPds2/MSEC.bin','rb'))
    MSEC.consensus_post(n_segments)
    pickle.dump(MSEC, open('results/AMPds2/MSEC.bin','wb'))

    del MSEC.modules[3]
    MSEC.num_modules -= 1
    MSEC.consensus_post(n_segments)
    pickle.dump(MSEC, open('results/AMPds2/MSEC_without_weather.bin','wb'))

    del MSEC.modules[2]
    MSEC.num_modules -= 1
    MSEC.consensus_post(n_segments)
    pickle.dump(MSEC, open('results/AMPds2/MSEC_without_weather_and_gas.bin','wb'))

    MSEC = pickle.load(open('results/AMPds2/MSEC.bin','rb'))
    del MSEC.modules[1]
    MSEC.num_modules -= 1
    MSEC.consensus_post(n_segments)
    pickle.dump(MSEC, open('results/AMPds2/MSEC_without_water.bin','wb'))

    del MSEC.modules[2]
    MSEC.num_modules -= 1
    MSEC.consensus_post(n_segments)
    pickle.dump(MSEC, open('results/AMPds2/MSEC_without_water_and_weather.bin','wb'))

def ampds_part2_consensus(n_segments=12):
    # HEre we load everything in and perform the consensus on each step.
    thresholds = [0.6, 0.8]
    MSEC = pickle.load(open('results/AMPds2/MSEC.bin','rb'))
    for threshold in thresholds:
        MSEC.threshold = threshold
        MSEC.consensus_post(n_segments)
        pickle.dump(MSEC, open(f'results/AMPds2/thresh={threshold}_MSEC.bin','wb'))

    del MSEC.modules[3]
    MSEC.num_modules -= 1
    for threshold in thresholds:
        MSEC.threshold = threshold
        MSEC.consensus_post(n_segments)
        pickle.dump(MSEC, open(f'results/AMPds2/thresh={threshold}_MSEC_without_weather.bin','wb'))

    del MSEC.modules[2]
    MSEC.num_modules -= 1
    for threshold in thresholds:
        MSEC.threshold = threshold
        MSEC.consensus_post(n_segments)
        pickle.dump(MSEC, open(f'results/AMPds2/thresh={threshold}_MSEC_without_weather_and_gas.bin','wb'))

    MSEC = pickle.load(open(f'results/AMPds2/thresh={threshold}_MSEC.bin','rb'))
    del MSEC.modules[1]
    MSEC.num_modules -= 1
    for threshold in thresholds:
        MSEC.threshold = threshold
        MSEC.consensus_post(n_segments)
        pickle.dump(MSEC, open(f'results/AMPds2/thresh={threshold}_MSEC_without_water.bin','wb'))

    del MSEC.modules[2]
    MSEC.num_modules -= 1
    for threshold in thresholds:
        MSEC.threshold = threshold
        MSEC.consensus_post(n_segments)
        pickle.dump(MSEC, open(f'results/AMPds2/thresh={threshold}_MSEC_without_water_and_weather.bin','wb'))

def fastdtw_wrapper(data):
    return np.array([[fastdtw(data[x], data[y])[0] for x in range(len(data))]for y in range(len(data))])

def canberra_wrapper(data):
    return pairwise_distances(data,metric='canberra')

def chebyshev_wrapper(data):
    return pairwise_distances(data,metric='chebyshev')

def mahalanobis_wrapper(data):
    return pairwise_distances(data,metric='mahalanobis')

def minkowski_wrapper(data):
    return pairwise_distances(data,metric='minkowski')

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

def f_measure(pred, true):
    value = (2 * len(pred & true)) / (len(true) + len(pred))
    return value

def eight_dim_investigation():
    seeds = [75]
    n_segments = 10
    dimension=8
    for seed in seeds:
        MSEC = pickle.load(open(f'results/rbfgenerator/8-dim/MSEC_seed_{seed}.bin','rb'))
        data = []
        for stream in range(12):
            data.append([pd.read_csv(f'data/rbfgenerator/{dimension}-dim/seed_{seed}_stream_{stream}_segment_{segment}_createdelete=False.csv') for segment in range(n_segments)])
            
        MSEC.modules = MSEC.modules[:6]+MSEC.modules[7:]+[MSEC.modules[6]]
        data = data[:6]+data[7:]+[data[6]]

        combined_data = []
        for segment in range(n_segments):
            combined_data.append(data[stream][segment].copy())
            for stream in range(1, 12):
                for i in range(dimension):
                    combined_data[segment][f'{i+stream*dimension}'] = data[stream][segment][f'{i}']
            cols = combined_data[segment].columns.tolist()
            cols = cols[:dimension]+cols[(dimension+1):]+[cols[dimension]]  # Swaps true cluster to end
            combined_data[segment] = combined_data[segment][cols]
            combined_data[segment].rename(columns={'cluster': 'true_cluster'}, inplace=True)

        
        for i in np.arange(1.2,5.1,0.1):
            evaluation_data_M = []
            for n_streams in range(2,12):
                MSEC.consensus_investigation(n_streams, n_segments, I=i)
                silhouette_scores_M = []
                F_scores_M = []
                num_clust_M = []

                for segment in range(n_segments):
                    combined_data[segment][f'MSI{n_streams}'] = MSEC.consensus_clusters_MC[segment]
                    D = pairwise_euclidean(combined_data[segment].iloc[:,:(dimension*n_streams)].to_numpy(copy=True))

                    F_scores_M.append(cluster_wise_f(combined_data[segment], cluster_column=f'MSI{n_streams}', true_cluster_column='true_cluster'))
                    num_clust_M.append(len(combined_data[segment][f'MSI{n_streams}'].unique()))
                    try:
                        silhouette_scores_M.append(silhouette_score(D, combined_data[segment][f'MSI{n_streams}'], metric='precomputed'))
                    except:
                        silhouette_scores_M.append(-11)

                evaluation_data_M.append(F_scores_M)
                evaluation_data_M.append(silhouette_scores_M)
                evaluation_data_M.append(num_clust_M)

            measures = ['F$_1$', 'SI', 'num_clusters']
            indices = pd.MultiIndex.from_tuples([(stream, measure) for stream in range(2, n_streams+1) for measure in measures])
            cols = columns=[f'$S^{{{x}}}$' for x in range(1,11)]
            M_df = pd.DataFrame(evaluation_data_M, columns=cols, index=indices)

            s = M_df.style
            with open(f'results/rbfgenerator/investigation/M_df_I={i}.tex', 'w') as f:
                f.write(s.to_latex(hrules=True))

        #evaluation_data.to_latex()
        
def main():
    # 2-dim
    rbfgenerator_tests(dimension=2, n_streams=3)
    rbf_generator_part2(dimension=2, n_streams=3, n_segments=10)

    # 8-dim
    rbfgenerator_tests(dimension=8, n_streams=12)
    rbf_generator_part2(dimension=8, n_streams=12, n_segments=10)
    #eight_dim_investigation()

    # AMPds2
    ampds_tests_baseline()
    ampds_tests_part1()
    ampds_part2_consensus(n_segments=12)
    #ampds_part2_removing_streams(n_segments=12)

if __name__ == '__main__':
    main()
