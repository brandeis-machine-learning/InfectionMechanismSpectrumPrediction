import copy
from random import choice
import os

import networkx as nx
import numpy as np
import csv
import sampling as sampling
from numpy import savetxt

from network.network import build_g
from network.network_data import list_of_hosts, list_of_viruses


# -------------------------------------------------------------------------------
# -----------------------------graph related-------------------------------------
# -------------------------------------------------------------------------------
def build_graph(bg):
    original_G_path = os.path.abspath('data/classifier/original_G.txt')
    # if network is not built, build the network
    if not os.path.exists(original_G_path) or bg:
        print('Building graph...')
        build_g(original_G_path=original_G_path, list_of_hosts=list_of_hosts,
                list_of_viruses=list_of_viruses)


def establish_training_G(G, run, fold, build_G):
    training_G_path = os.path.abspath('data/classifier/training_G.txt')
    ith_fold_path = os.path.abspath(
        'data/evaluation/cross_validation/run-' + str(run) + '-fold-' + str(fold) + '-X.csv')
    ith_fold = []
    with open(ith_fold_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            ith_fold.append(row)
    if build_G:
        # remove edges in the ith_fold --> use the remaining folds for training
        for edge in ith_fold:
            G.remove_edge(edge[0], edge[1])
        # save training graph
        nx.write_gml(G, training_G_path)
    # return training_G_path
    num_of_test_edges = len(ith_fold)
    return training_G_path, num_of_test_edges


def load_graph(training_G_path, structural_emb_path):
    new_G = nx.read_gml(training_G_path)

    structural_emb_dict = load_node_embeddings(structural_emb_path)

    return new_G, structural_emb_dict


# -------------------------------------------------------------------------------
# --------------------------classifier related-----------------------------------
# -------------------------------------------------------------------------------
def load_node_embeddings(emb_file_path):
    node_dict = {}
    with open(emb_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        info = next(csv_reader)
        count = int(info[0])
        vec = next(csv_reader)
        for i in range(0, count):
            temp = [float(numeric_string) for numeric_string in vec]
            node_dict[int(temp[0])] = temp[1:]
            if i == count - 1:
                break
            vec = next(csv_reader)
    return node_dict


def load_training_data(full_G, training_G, structural_emb_dict, content_emb_dict, num_of_test_edges):
    X = np.empty([len(structural_emb_dict) * (len(structural_emb_dict) - 1),
                  2 * (len(structural_emb_dict[1])) + len(content_emb_dict[(0, 1)])])

    y = np.empty([len(structural_emb_dict) * (len(structural_emb_dict) - 1), 1])
    y_copy = np.empty([len(structural_emb_dict) * (len(structural_emb_dict) - 1), 1])
    count = 0
    for i in range(0, len(structural_emb_dict)):
        for j in range(0, len(structural_emb_dict)):
            if i != j:
                arr_i = np.array(structural_emb_dict[i])
                arr_j = np.array(structural_emb_dict[j])

                y_to_add = get_y_to_add(full_G, i, j)
                y_to_add_copy = get_y_to_add(training_G, i, j)

                # add embedding of the two nodes to represent edge
                structural_vec = np.concatenate((arr_i, arr_j))
                arr_i_j = np.array(content_emb_dict[(i, j)])
                edge_to_add = np.concatenate((structural_vec, arr_i_j))
                X[count] = edge_to_add
                y[count] = y_to_add
                y_copy[count] = y_to_add_copy
                count = count + 1
    # save np matrix to file
    savetxt(os.path.abspath('data/classifier/X.txt'), X)
    savetxt(os.path.abspath('data/classifier/y.txt'), y)
    savetxt(os.path.abspath('data/classifier/y_copy.txt'), y_copy)

    index2pair_dict = get_index2pair_dict(len(X), full_G)
    X_train, y_train, X_test_negatives, y_test_negatives, all_selected_indices \
        = sampling.random_sampling(training_G, X, y, y_copy,
                                   num_of_training_edges=len(training_G.edges()),
                                   num_of_test_edges=num_of_test_edges, index2pair_dict=index2pair_dict)

    savetxt(os.path.abspath('data/classifier/X_train.txt'), X_train)
    savetxt(os.path.abspath('data/classifier/y_train.txt'), y_train)

    return X_train, y_train, X_test_negatives, y_test_negatives, all_selected_indices, index2pair_dict


def get_y_to_add(full_G, i, j):
    if (str(i), str(j)) in full_G.edges():
        edge_relation = full_G.edges[str(i), str(j)]['relation']
        if edge_relation.__contains__('similar'):
            y_to_add = np.array([1])
        elif edge_relation.__contains__('infects'):
            y_to_add = np.array([2])
        elif edge_relation.__contains__('belongs'):
            y_to_add = np.array([3])
        else:
            y_to_add = np.array([4])
    else:
        y_to_add = np.array([0])
    return y_to_add


def get_index2pair_dict(length, G):
    index2pair_dict = {}
    src_node = 0
    dst_node = 0
    count = 0
    num_of_nodes = len(G.nodes())
    # fill in the dict
    while count < length:
        if src_node == dst_node:
            dst_node = (dst_node + 1) % num_of_nodes
            continue
        # add new node pair to the dict
        index2pair_dict[count] = (src_node, dst_node)
        dst_node = dst_node + 1
        if dst_node == num_of_nodes:
            src_node = src_node + 1
            dst_node = 0
        count = count + 1
    return index2pair_dict


def load_test_data(full_G, test_set_positives, structural_emb_dict, content_emb_dict, X_negatives, y_negatives):
    size_of_positive_test_set = len(test_set_positives)

    X_positives = np.empty(
        [2 * size_of_positive_test_set, 2 * len(structural_emb_dict[1]) + len(content_emb_dict[(0, 1)])])

    y_positives = np.empty([2 * size_of_positive_test_set, 1])

    count = 0
    for pair in test_set_positives:
        i = int(pair[0])
        j = int(pair[1])
        # construct y_test
        arr_i = np.array(structural_emb_dict[i])
        arr_j = np.array(structural_emb_dict[j])
        arr_i_j = np.array(content_emb_dict[(i, j)])
        edge_to_add_1 = np.concatenate((arr_i, arr_j, arr_i_j))
        edge_to_add_2 = np.concatenate((arr_j, arr_i, arr_i_j))
        X_positives[count] = edge_to_add_1
        X_positives[count + 1] = edge_to_add_2

        # construct y_test
        if (str(i), str(j)) in full_G.edges():
            edge_relation = full_G.edges[str(i), str(j)]['relation']
            if edge_relation.__contains__('similar'):
                y_to_add = np.array([1])
            elif edge_relation.__contains__('infects'):
                y_to_add = np.array([2])
            elif edge_relation.__contains__('belongs'):
                y_to_add = np.array([3])
            else:
                y_to_add = np.array([4])
        else:
            y_to_add = np.array([0])
        # save
        y_positives[count] = y_to_add
        y_positives[count + 1] = y_to_add
        # increment count
        count = count + 2

    X_test = np.vstack([X_positives, X_negatives])
    y_test = np.concatenate([y_positives.flatten(), y_negatives.flatten()])

    savetxt(os.path.abspath('data/classifier/X_test.txt'), X_test)
    savetxt(os.path.abspath('data/classifier/y_test.txt'), y_test)

    return X_test, y_test


# -------------------------------------------------------------------------------
# -----------------------------save prediction-----------------------------------
# -------------------------------------------------------------------------------
def filter_PPI_pred(G, edge_type, binding, emb_name):
    existed = remove_duplicates(emb_name + '_' + edge_type)
    with open(os.path.abspath(
            'data/prediction/prediction_' + emb_name + '_' + edge_type + '.csv'), 'a') as file:
        for e in G.edges():
            edge_data = G.get_edge_data(*e)
            src = str(e[0])
            dst = str(e[1])
            if edge_data['relation'].__contains__('interacts') and edge_data['etype'] == 'predicted':
                if not ((src, dst) in existed or (dst, src) in existed) and \
                        G.nodes[src]['group'] != G.nodes[dst]['group'] and \
                        'protein' in G.nodes[src]['group'] and 'protein' in G.nodes[dst]['group']:
                    basic_info = src + ',' + dst + ',' + G.nodes[src]['type'] + ' ' + \
                                 G.nodes[src]['host'] + ',' + G.nodes[dst]['type'] + ' ' + \
                                 G.nodes[dst]['host'] + ',' + str(edge_data['probability_estimate']) + ',' + \
                                 edge_data['connection'] + ','
                    if (G.nodes[src]['type'] == 'Spike' and
                        G.nodes[dst]['type'] == 'ACE2') or \
                            (G.nodes[src]['type'] == 'ACE2' and
                             G.nodes[dst]['type'] == 'Spike') or \
                            (G.nodes[src]['type'] == 'DPP4' and
                             G.nodes[dst]['type'] == 'Spike') or \
                            (G.nodes[src]['type'] == 'Spike' and
                             G.nodes[dst]['type'] == 'DPP4'):
                        to_write = basic_info + 'interacts' + ',' + 'likely' + '\n'

                        binding.append((G.nodes[src]['host'] + ' ' + G.nodes[dst]['host']))
                        binding.append((G.nodes[dst]['host'] + ' ' + G.nodes[src]['host']))

                    elif G.nodes[src]['type'] == 'Spike' or G.nodes[dst]['type'] == 'Spike':
                        to_write = basic_info + 'interacts' + ',' + 'unlikely' + '\n'
                    else:
                        to_write = basic_info + 'interacts' + ',' + 'likely' + '\n'
                    file.write(to_write)
            elif edge_data['relation'].__contains__('interacts') and edge_data['etype'] == 'original':
                binding.append((G.nodes[src]['host'] + ' ' + G.nodes[dst]['host']))
                binding.append((G.nodes[dst]['host'] + ' ' + G.nodes[src]['host']))
        file.close()


def filter_infection_pred(G, edge_type, emb_name):
    existed = remove_duplicates('temp_' + emb_name + '_' + edge_type)
    with open(os.path.abspath(
            'data/prediction/prediction_temp_' + emb_name + '_' + edge_type + '.csv'),
            'a') as file:
        for e in G.edges():
            edge_data = G.get_edge_data(*e)
            if edge_data['relation'].__contains__('infects') and edge_data['etype'] == 'predicted':
                src = str(e[0])
                dst = str(e[1])
                if not ((src, dst) in existed or (dst, src) in existed) and \
                        G.nodes[src]['group'] != G.nodes[dst]['group']:
                    if ((G.nodes[src]['type'] == 'virus' and G.nodes[dst]['type'] == 'host')
                            or (G.nodes[src]['type'] == 'host' and G.nodes[dst]['type'] == 'virus')):
                        basic_info = src + ',' + dst + ',' + \
                                     G.nodes[src]['type'] + ' ' + \
                                     G.nodes[src]['host'] + ',' + \
                                     G.nodes[dst]['type'] + ' ' + \
                                     G.nodes[dst]['host'] + ',' + \
                                     str(edge_data['probability_estimate']) + ',' + \
                                     edge_data['connection'] + ',' + \
                                     'infects'
                        basic_info = basic_info + '\n'
                        file.write(basic_info)
        file.close()


def filter_unlikely_inf(binding, emb_name):
    with open(os.path.abspath('data/prediction/prediction_temp_' + emb_name + '_infects.csv'), 'r') as read_csv, \
            open(os.path.abspath('data/prediction/prediction_' + emb_name + '_infects.csv'), 'w') as write_csv:
        csv_reader = csv.reader(read_csv, delimiter=',')
        for row in csv_reader:
            virus = row[2].split(' ', 1)[1]
            host = row[3].split(' ', 1)[1]
            token = virus + ' ' + host
            if binding.__contains__(token):
                to_write = \
                    str(row).replace(', ', ',').replace('\'', '').replace('[', '').replace(']', '').rsplit(',', 1)[0] \
                    + ',likely' + '\n'
                write_csv.write(to_write)
            else:
                to_write = \
                    str(row).replace(', ', ',').replace('\'', '').replace('[', '').replace(']', '').rsplit(',', 1)[0] \
                    + ',unlikely' + '\n'
                write_csv.write(to_write)
        write_csv.close()
    os.remove(os.path.abspath('data/prediction/prediction_temp_' + emb_name + '_infects.csv'))


def remove_duplicates(edge_type):
    existed = []
    # read all existed predictions --> eliminate duplicates
    if (os.path.exists(os.path.abspath(
            'data/prediction/prediction_' + edge_type + '.csv'))):
        with open(os.path.abspath(
                'data/prediction/prediction_' + edge_type + '.csv'), 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                existed.append((str(row[0]), str(row[1])))
    return existed


def build_graph_alt(graph_path):
    # for leave-one-out for Sus scrofa, no need to
    names = ['Hom_sap', 'Fel_cat', 'Mac_mul', 'Can_lup', 'Rhi_fer', 'Mes_aur', 'Sus_scr']
    virus_node = 82
    host_nodes = [204, 209, 216, 207, 219, 220]
    for i in range(len(names)):
        graph = nx.read_gml(graph_path)
        # for sus scrofa, no need to remove because it is not added to the original G as ground truth
        if names[i] != 'Sus_scr':
            graph.remove_edge(str(virus_node), str(host_nodes[i]))
        nx.write_gml(graph, os.path.abspath('data/classifier/original_G_' + names[i] + '.txt'))
    return names

# build_graph_alt(os.path.abspath('data/classifier/original_G.txt'))
