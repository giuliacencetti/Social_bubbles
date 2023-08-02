import networkx as nx
import numpy as np
import pickle
#import pickle5 as pickle
import os.path as osp
import os
from tqdm import tqdm_notebook as tqdm
from load_temporal_graph import load_df_rssi_filter, build_graphs, get_array_of_contacts




def temporal_clustered_network_generation(n_time_steps, nb_clusters, cluster_size, p_intra, p_inter):
    graphs = []
    for time_step in tqdm(range(n_time_steps)):
        G = clustered_network_generation(nb_clusters,cluster_size,p_intra,p_inter)
#         while nx.is_connected(G) == False:
#             G = clustered_network_generation(nb_clusters,cluster_size,p_intra,p_inter)
        graphs.append(G)

    return graphs


def clustered_network_generation(nb_clusters,cluster_size,p_intra,p_inter):
    sizes = [cluster_size for n in range(nb_clusters)]
    # P is a matrix of size nb_clusters where element (i,j) is the edge probability between cluster i and cluster j
    # I put p_intra on the diagonal and p_inter out-diagonal
    P = np.array([[p_inter for x in range(nb_clusters)] for y in range(nb_clusters)])
    np.fill_diagonal(P, p_intra)
    G = nx.stochastic_block_model(sizes, P)

    return G


def temporal_clustered_network_generation_diff_sizes(n_time_steps, sizes,p_intra,p_inter):
    graphs = []
    for time_step in tqdm(range(n_time_steps)):
        G = clustered_network_generation_diff_sizes(sizes,p_intra,p_inter)
#         while nx.is_connected(G) == False:
#             G = clustered_network_generation(nb_clusters,cluster_size,p_intra,p_inter)
        graphs.append(G)

    return graphs


def clustered_network_generation_diff_sizes(sizes,p_intra,p_inter):
    # P is a matrix of size nb_clusters where element (i,j) is the edge probability between cluster i and cluster j
    # I put p_intra on the diagonal and p_inter out-diagonal
    P = np.array([[p_inter for x in range(len(sizes))] for y in range(len(sizes))])
    np.fill_diagonal(P, p_intra)
    G = nx.stochastic_block_model(sizes, P)

    return G


def temporal_graph_load(name):
    path = 'Graphs/' + name + '/'

    with open(path + 'num_snapshots.txt', 'r') as input:
        n_graphs = int(float(input.read()))

    graphs = []
    for idx in range(n_graphs):
        with open(path + 'graph_%d.pkl' % idx, 'rb') as input:
            graph = pickle.load(input)
            graphs.append(graph)
    return graphs


def temporal_graph_len(name):
    path = 'Graphs/' + name + '/'

    with open(path + 'num_snapshots.txt', 'r') as input:
        n_graphs = int(float(input.read()))

    graphs = []
    for idx in range(n_graphs):
        with open(path + 'graph_%d.pkl' % idx, 'rb') as input:
            graph = pickle.load(input)
            graphs.append(graph)
    graph_len = len(graphs)
    return graph_len


def temporal_graph_save(graphs, name):
    path = 'Graphs/' + name + '/'
    if not osp.exists(path):
        os.makedirs(path)
    with open(path + 'num_snapshots.txt', 'w') as output:
        output.write('%f' % len(graphs))
    for idx, graph in enumerate(graphs):
        with open(path + 'graph_%d.pkl' % idx, 'wb') as output:
            pickle.dump(graph, output, pickle.HIGHEST_PROTOCOL)


def get_individuals_from_graphs(graphs):
    """
    Get the individuals who are present in the list of graphs.

    The function return a list of all the unique individuals who are present in
    the list of snapshots of the temporal graph.

    Parameters
    ----------
    graphs: list
        snapshots of a temporal graph

    Returns
    ----------
    nodes_list: list
        individuals in the list of graphs
    """

    nodes_list = []
    for g in graphs:
        nodes_list.extend(list(g.nodes()))
    nodes_list = np.unique(nodes_list)

    return nodes_list




def get_DTU_graph_rssi_filter(temporal_gap, rssi_filter, n_individuals=None, n_row=None):
    name = 'DTU'
    csv_file = '../../../covid_isolation_project/new/Dataset/bt_symmetric.csv'
    graphs = get_graph_from_csv_rssi_filter(name, csv_file, temporal_gap, rssi_filter, n_individuals, n_row)

    return graphs


def temporal_graph_save(graphs, name):
    path = 'Graphs/' + name + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'num_snapshots.txt', 'w') as output:
        output.write('%f' % len(graphs))
    for idx, graph in enumerate(graphs):
        with open(path + 'graph_%d.pkl' % idx, 'wb') as output:
            pickle.dump(graph, output, pickle.HIGHEST_PROTOCOL)


def temporal_graph_load(name):
    path = 'Graphs/' + name + '/'

    with open(path + 'num_snapshots.txt', 'r') as input:
        n_graphs = int(float(input.read()))

    graphs = []
    for idx in range(n_graphs):
        with open(path + 'graph_%d.pkl' % idx, 'rb') as input:
            graph = pickle.load(input)
            graphs.append(graph)
    return graphs


def get_graph_from_csv_rssi_filter(name, csv_file, temporal_gap, rssi_filter, n_individuals=None, n_row=None):
    name += '_temporal_gap_%.0f_rssi_filter_%d' % (temporal_gap, rssi_filter)

    if os.path.exists('Graphs/' + name + '/'):
        print('Graph already computed: load from memory')
        graphs = temporal_graph_load(name)
    else:
        print('Graph not already computed: build from data')
        df = load_df_rssi_filter(csv_file, rssi_filter, n_individuals=n_individuals, n_row=n_row)
        graphs = build_graphs(get_array_of_contacts(df, temporal_gap, column_name='# timestamp'),
                                                    temporal_gap)
        temporal_graph_save(graphs, name)
    return graphs
