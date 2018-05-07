import pandas as pd
import os
import networkx as nx
import DataCleaning as dc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.misc import factorial
import NetworkAnalysis as na

"""
ASSEMBLING THE NETWORK GRAPH DICTIONARY
"""
column_names = ['node1','node2','time','qvote','avote','veri']
ecol = pd.read_csv('eigen_complete_ordered_list.txt',' ',header=None)
ecol.columns

ecol.columns = column_names
ecol.shape
undirected = nx.Graph()

#Creating Graph
for edge in range(len(ecol)):
    node1 = ecol.iloc[edge,0]
    node2 = ecol.iloc[edge,1]
    time  = ecol.iloc[edge,2]
    qvote = ecol.iloc[edge,3]
    avote = ecol.iloc[edge,4]
    veri  = ecol.iloc[edge,5]
    #Ausschluss von Self loops
    #Parallele Edges werden zugelassen
    if undirected.has_edge(node1,node2) == True:
        continue
    if node1 != node2:
        undirected.add_edge(node1,node2, time=time, qvote=qvote, avote=avote, veri=veri)

len(undirected.edges())
len(ecol)
ms_in_week = 604800000
mintime = 1279207934383

#latest time from DataFrame
maxi = ecol['time'].max()
iter_range = 1+int((maxi-mintime)/ms_in_week)
iter_range

list_of_graphs = []

for i in range(iter_range):
    j = i+1
    maxtime = 1279207934383 + j*ms_in_week
    #Filtering the Graph by time with Max's fancy filter function
    proxi = dc.filter_network_attributes(undirected, mintime, maxtime, -1, -1, -1, -1, -1)
    list_of_graphs.append(proxi)

len(list_of_graphs)
type(list_of_graphs[5])
list_of_graphs[2]



"""
Calculating Network Attributes for each of the time stepped networks in the list
"""

"""
BASIC ATTRIBUTES OVER TIME
"""
#Dataframe containing all the basic ATTRIBUTES

convert_array = []
for w in range(len(list_of_graphs)):
    current_network = list_of_graphs[w]

    Number_of_Nodes                = len(current_network.nodes())
    Number_of_Edges                = len(current_network.edges())
    Number_of_Connected_Components = nx.number_connected_components(current_network)
    #Number_of_Self_Loops           = current_network.number_of_selfloops()

    comps = list(nx.connected_component_subgraphs(list_of_graphs[w]))
    max_comp = max(comps, key=len)
    Size_of_Giant_Component        = len(max_comp.nodes())

    convert_array.append([Number_of_Nodes,Number_of_Edges,Number_of_Connected_Components,Size_of_Giant_Component])


basic_attributes_ot = pd.DataFrame(convert_array, columns= ['Number of Nodes','Number of Edges','Number of Connected Components','Size of Giant Component'])
#basic_attributes_ot.head()

"""
DEGREE
"""
#Average Degree over Time
list_of_ave_k = []

for t in range(len(list_of_graphs)):
    liste = list(list_of_graphs[t].degree(list_of_graphs[t].nodes()))
    degreesum = sum([pair[1] for pair in liste])
    ave_k = degreesum/len(list_of_graphs[t].nodes())
    list_of_ave_k.append(ave_k)


#Dictionary with entries for all timesteped networks containing a list of the degree for each node
dict_of_node_degrees_for_all_graphes = {}

for t in range(len(list_of_graphs)):
    iter_graph = list_of_graphs[t]
    iter_graph_node_list = list(iter_graph.nodes())
    length = len(iter_graph_node_list)
    node_degree_list = []

    for h in range(length):
        iter_node = iter_graph_node_list[h]
        node_degree_list.append(iter_graph.degree(iter_node))

    node_degree_list
    dict_of_node_degrees_for_all_graphes[t] = node_degree_list


"""
# degree distribution for each entry of dict_of_node_degrees_for_all_graphes
# same integer binning for the histogramms so all timesteeps can easily compared
# the number of bins is derived from the max degree in the last time stepped
"""

"""
CLUSTERING COEFFICIENT
"""
clustering_coefficients=[]
for q in range(len(list_of_graphs)):
    trip = nx.average_clustering(list_of_graphs[q])
    clustering_coefficients.append(trip)

len(clustering_coefficients)

################################################################################
"""
Comparing our Data to Network Evolution Models

 - Preferential Attachment
 - Growth Only

 Issues: Structual Evolution vs Temporal Evolution

"""

#When do nodes appear? When exacty does their degree grow?
#First, what are the ids of all the unique user nodes?
#node1 = ecol.iloc[0], node2, time, qvote, avote, veri
un1 = ecol.node1.unique()
un2 = ecol.node2.unique()
node_id_proxi = np.append(un1,un2)

node_ids = np.unique(node_id_proxi)
len(node_ids)
node_ids
#when was each node added to the the network and when did the gain connections?
node_history_dict = {}

for unique in node_ids:
    uni = str(unique)
    node_history_dict[uni] = []

for z in range(len(ecol)):
    node1 = ecol.iloc[z,0]
    node2 = ecol.iloc[z,1]
    time  = ecol.iloc[z,2]
    if undirected.has_edge(node1,node2) == True:
        if time != undirected.edges[node1,node2]['time']:
            continue
    if node1 != node2:
        nd1 = str(node1)
        nd2 = str(node2)
        #hier ist das PROBLEEEEM!
        node_history_dict[nd1].append(time)
        node_history_dict[nd2].append(time)


# len(node_history_dict[1641621])
# undirected.degree(1641621)
# undirected[1641621]
node_history_dict
#birth time list of the Nodes
birth_times = []

type(node_history_dict[1641621])
wtf = list(node_history_dict.keys())

#list(node_history_dict.keys())
for x in wtf:
    what = str(x)
    click = node_history_dict[what]
    birth_times.append(click)

birth_times

type(int(x))
"""
PLOTs

To Dos:
    Growth of the Network:
    - Number of Nodes over Time
    - Average Degree etc.
    - p(k) over k plots


"""

extractmaxdegree = list(list_of_graphs[371].degree(list_of_graphs[371].nodes()))
degrees = [click[1] for click in extractmaxdegree]
# max degree in final network graph
num_bins = max(degrees)
num_bins
# this will be the number of bins. The bin height will represent the count of nodes with the respective degree.

#data from dict_of_node_degrees_for_all_graphes with index from 0 to 371
# for u in range(len(list_of_graphs)):
#     iter_node_degree_list = dict_of_node_degrees_for_all_graphes[u]

iter_node_degree_list = dict_of_node_degrees_for_all_graphes[371]
# the bins should be of integer width, because poisson is an integer distribution
entries, bin_edges, patches = plt.hist(iter_node_degree_list, bins=num_bins, range=[-0.5, num_bins-0.5], normed=True)

# calculate binmiddles
bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])

# poisson function, parameter lamb is the fit parameter
def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

# fit with curve_fit
parameters, cov_matrix = curve_fit(poisson, bin_middles, entries)

# plot poisson-deviation with fitted parameter
x_plot = np.linspace(0, 20, 1000)

plt.plot(x_plot, poisson(x_plot, *parameters), 'r-', lw=2)
plt.show()
