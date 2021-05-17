"""import modules"""
import csv
import numpy as np
import os as os
from network import network_data as graph_data


# -----------------------------------------------------
# ---------------------Preparation---------------------
# -----------------------------------------------------

def get_index_name_map(file_name):
    """
    build the dictionary of {'index on axis': 'name of the element'}
    :param file_name: name of the file to be read
    :return: a dict of all nodes which maps axis index to node name
    """
    # """convert the map to an index-protein_name map"""
    nodes_dic = {}
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        nodes_list = next(csv_reader)
        for index in range(1, len(nodes_list)):
            nodes_dic[index - 1] = nodes_list[index]
        return nodes_dic


def read_similarity_matrix(file_name, end_len):
    """
    read the network in the file
    :param file_name: name of the file to be read
    :param end_len: the end length
    :return: a similarity matrix
    """
    matrix = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        value_list = next(csv_reader)
        for i in range(0, end_len):
            temp = []
            for index in range(1, len(value_list)):
                temp.append(value_list[index])
            arr = np.array(temp)  # convert to np array
            if len(matrix) == 0:  # first insertion into the matrix
                matrix = np.array(arr)
            else:
                matrix = np.vstack([matrix, arr])
            if i == end_len - 1:
                break
            value_list = next(csv_reader)
    return matrix


# -----------------------------------------------------
# -------------------Initialize Graph------------------
# -----------------------------------------------------

def build_one_homo_graph(G, file_name, protein_name, group_name, belong_relation_dict, node_index):
    """
    build one homo network and merge into G
    :param belong_relation_dict: ...
    :param G: network G
    :param file_name: ...
    :param protein_name: protein name read from file name
    :param group_name: layer name read from file name
    :param node_index: ...
    :return: two list of added nodes and edges
    """
    index_name_map = get_index_name_map(file_name)
    similarity_matrix = read_similarity_matrix(file_name, len(index_name_map))
    return init_homo_graph(G, index_name_map, similarity_matrix, protein_name, group_name, belong_relation_dict,
                           node_index)


def init_homo_graph(G, index_name_map, similarity_matrix, protein_name, group_name, belong_relation_dict, node_index):
    """
    initiate the DiGraph network using networkx, put nodes and edges in the homo network
    :param node_index:
    :param belong_relation_dict: ...
    :param G: network G
    :param index_name_map: ...
    :param similarity_matrix: ...
    :param protein_name: ...
    :param group_name: ...
    :return: two list of added nodes and edges
    """

    # initialize node and edge lists
    node_list = []
    edge_list = []
    start_index = node_index

    # set up nodes
    for idx in index_name_map.keys():
        name = index_name_map.get(idx)
        # host is in '[]' in the protein's name
        host = name

        inner_list = belong_relation_dict[host]
        if not (protein_name == 'host' or protein_name == 'virus'):
            inner_list.append(protein_name)
        # 1) type is its protein name
        # 2) host is the name of its animal host for host proteins; name of the virus for virus proteins
        if group_name == 'virus' or group_name == 'host':
            G.add_node(str(node_index), disp=host, type=protein_name, group=group_name,
                       host=host)
        else:
            G.add_node(str(node_index), disp=(protein_name + ', ' + host), type=protein_name, group=group_name,
                       host=host)
        # append added nodes to node_list
        node_list.append(str(node_index))
        node_index = node_index + 1

    # set up edges
    for row in range(0, len(similarity_matrix)):
        for col in range(0, row + 1):
            # similarity between a protein and its own not meaningful
            if similarity_matrix[row][col] == "null" or row == col:
                continue
            G.add_edge(str(start_index + row), str(start_index + col), similarity=similarity_matrix[row][col],
                       relation='similar', category='homogeneous', etype='original')
            # append added edges to edge_list
            edge_list.append((str(start_index + row), str(start_index + col)))

    return node_list, edge_list, node_index


def build_original_hetero_edges(G, hetero_edges, dict_of_belong_relations, dict_of_nodes_groups, list_of_virus):
    # add host-host protein edges and virus-virus protein edges
    for key in dict_of_belong_relations:
        is_host = True
        if key in list_of_virus:
            is_host = False
        hetero_edges = hetero_edges + (add_hetero_edges(G=G,
                                                        dict_of_nodes_groups=dict_of_nodes_groups,
                                                        group_1='host' if is_host is True else 'virus',
                                                        type_1=['host'] if is_host is True else ['virus'],
                                                        host_list_1=[key],
                                                        group_2='host protein' if is_host is True else 'virus protein',
                                                        type_2=dict_of_belong_relations[key],
                                                        host_list_2=[key],
                                                        relation='belongs',
                                                        etype='original')
                                       )

    # add virus protein to host protein edges and virus to host edges
    for info_dict in graph_data.hetero_edge_data:
        hetero_edges = hetero_edges + (add_hetero_edges(G=G,
                                                        dict_of_nodes_groups=dict_of_nodes_groups,
                                                        group_1=info_dict['group_1'],
                                                        type_1=info_dict['type_1'],
                                                        host_list_1=info_dict['host_list_1'],
                                                        group_2=info_dict['group_2'],
                                                        type_2=info_dict['type_2'],
                                                        host_list_2=info_dict['host_list_2'],
                                                        relation=info_dict['relation'],
                                                        etype='original'))


def build_home_graph(G, dict_of_nodes_groups, dict_of_edges_groups, file_dir, belong_relation_dict):
    """
    build homogeneous network into the network G
    :param belong_relation_dict: ...
    :param G: Graph G
    :param dict_of_nodes_groups: {'group':         {'type'  : ['node']}},
                            e.g. {'virus protein': {'Spike' : [Spike_0, Spike_1,...]}}
    :param dict_of_edges_groups: {'group':         {'type': ['edge']}}
                            e.g. {'virus protein': {'Spike' : [(Spike_0, Spike_1), (Spike_0, Spike_2)...]}}
    :param file_dir: the directory that contains all csv files
    """

    # initialize file iterator
    node_index = 0
    entries = os.scandir(file_dir)
    # iterate through all the file entries
    for file in entries:
        # get the layer and type from the file name
        meta_info = file.name.split('.')[0].rsplit('-', 1)
        protein_name = meta_info[0]
        group_name = meta_info[1]

        # get added nodes as a list of str, added edges as a list of 2-tuple
        node_list, edge_list, node_index = build_one_homo_graph(G=G,
                                                                file_name=file,
                                                                protein_name=protein_name,
                                                                group_name=group_name,
                                                                belong_relation_dict=belong_relation_dict,
                                                                node_index=node_index)

        # update the dict
        dict_of_nodes_groups[group_name][protein_name] = node_list
        dict_of_edges_groups[group_name][protein_name] = edge_list


def add_hetero_edges(G, dict_of_nodes_groups, group_1, type_1, host_list_1, group_2, type_2, host_list_2, relation,
                     etype):
    """
    add heterogeneous edges to the network
    :param etype: edge type, predicted or original
    :param classifier: network associated with the edge
    :param G: network G
    :param dict_of_nodes_groups:
    :param group_1: layer attr of src node
    :param type_1: type attr of src node
    :param host_list_1: host attr of src node
    :param group_2: layer attr of dst node
    :param type_2: type attr of dst node
    :param host_list_2: host attr of dst node
    :return:
    """
    hetero_edges = []
    # iterate through nodes in the correct layer and correct type
    # fill the list: hetero_edges
    for first_type in type_1:
        # first_type = ORF3b
        # dict_of_nodes_groups[group_1][first_type] is a list of ORF3b nodes
        # nd_1 is a node in the list
        for nd_1 in dict_of_nodes_groups[group_1][first_type]:
            # get the node in the network at key 'nd_1', which returns dict of all attributes of the node
            # then, match the host
            if G.nodes[nd_1]['host'] in host_list_1:
                # second_type = IRF3
                for second_type in type_2:
                    # dict_of_nodes_groups[group_2][second_type] = list of IRF3 nodes
                    # nd_2 is a IRF3 node
                    for nd_2 in dict_of_nodes_groups[group_2][second_type]:
                        if G.nodes[nd_2]['host'] in host_list_2:
                            hetero_edges.append((nd_1, nd_2))

    # add edges to the network
    for ele in hetero_edges:
        G.add_edge(str(ele[0]), str(ele[1]), relation=relation, category='heterogeneous', etype=etype)
    return hetero_edges
