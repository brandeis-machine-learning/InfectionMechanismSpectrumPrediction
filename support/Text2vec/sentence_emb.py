import networkx as nx
import os as os
import pickle
from network.network_data import protein_function_data
from utils import get_index2pair_dict

from support.Text2vec.text2vec import Text2vec


def edge_content_emb():
    original_G_path = os.path.abspath('data/classifier/original_G.txt')
    full_G = nx.read_gml(original_G_path)
    # returns a dict of {node: attr}
    node_host = nx.get_node_attributes(full_G, 'host')
    node_group = nx.get_node_attributes(full_G, 'group')
    node_type = nx.get_node_attributes(full_G, 'type')
    edge_list_attr = []
    # construct sentences for pairs
    for src_node in full_G.nodes():
        for dst_node in full_G.nodes():
            if src_node != dst_node:
                basic_info = node_host[src_node] + ' ' + node_group[src_node] + ' ' + node_type[src_node] + ' ' + \
                             node_host[dst_node] + ' ' + node_group[dst_node] + ' ' + node_type[dst_node] + ' ' + \
                             protein_function_data[node_type[src_node]] + ' ' + \
                             protein_function_data[node_type[dst_node]]

                if (str(src_node), str(dst_node)) in full_G.edges():
                    to_add = basic_info + ' ' + full_G[src_node][dst_node]['relation']
                else:
                    if node_type[src_node] == node_type[dst_node]:
                        to_add = basic_info + ' similar'
                    elif (node_group[src_node] == 'virus' and node_group[dst_node] == 'host') or (
                            node_group[src_node] == 'host' and node_group[dst_node] == 'virus'):
                        to_add = basic_info + ' infects'
                    elif (node_group[src_node] == 'virus' and node_group[dst_node] == 'virus protein') or (
                            node_group[src_node] == 'virus protein' and node_group[dst_node] == 'virus'):
                        to_add = basic_info + ' belongs'
                    elif (node_group[src_node] == 'host' and node_group[dst_node] == 'host protein') or (
                            node_group[src_node] == 'host protein' and node_group[dst_node] == 'host'):
                        to_add = basic_info + ' belongs'
                    elif (node_group[src_node] == 'host protein' and node_group[dst_node] == 'virus protein') or (
                            node_group[src_node] == 'virus protein' and node_group[dst_node] == 'host protein'):
                        to_add = basic_info + ' interacts'
                    else:
                        to_add = basic_info + ' unconnected'
                edge_list_attr.append(to_add)

    # extract index -> node pair dict
    index2pair_dict = get_index2pair_dict(len(edge_list_attr), G=full_G)

    # preprocess text2vec model, convert list to list to tokens
    t2v = Text2vec(edge_list_attr)

    # TF-IDF weighted Glove vector summary for document list
    # Input: a list of documents
    # Output: Matrix of vector for all the documents
    docs_emb = t2v.tfidf_weighted_wv()

    # store embedding results to a dict
    edge_emb_dict = {}
    for n in range(len(docs_emb)):
        node_pair = index2pair_dict[n]
        edge_emb_dict[node_pair] = docs_emb[n]

    print('Saving edge content embedding to file...')
    # save the dict to disk
    with open(os.path.abspath('data/embedding/sentence_embedding/sentence_embedding.pkl'), 'wb') as file:
        pickle.dump(edge_emb_dict, file)
        file.close()


def node_content_emb():
    original_G_path = os.path.abspath('data/classifier/original_G.txt')
    full_G = nx.read_gml(original_G_path)
    # returns a dict of {node: attr}
    node_host = nx.get_node_attributes(full_G, 'host')
    node_group = nx.get_node_attributes(full_G, 'group')
    node_type = nx.get_node_attributes(full_G, 'type')
    list_attr = []
    for node in full_G.nodes():
        to_add = node_host[node] + ' ' + node_group[node] + ' ' + node_type[node] + ' ' + protein_function_data[
            node_type[node]]
        list_attr.append(to_add)

    # preprocess text2vec model, convert list to list to tokens
    t2v = Text2vec(list_attr)

    # Input: a list of documents, Output: Matrix of vector for all the documents
    docs_emb = t2v.tfidf_weighted_wv()

    # store embeddings in a dict
    node_emb_dict = {}
    idx = 0

    for emb in docs_emb:
        node_emb_dict[str(idx)] = emb
        idx = idx + 1

    with open(os.path.abspath(
            'data/embedding/sentence_embedding/sentence_embedding_node.pkl'), 'wb') as file:
        pickle.dump(node_emb_dict, file)
        file.close()


# if __name__ == '__main__':
#     node_content_emb()
#     edge_content_emb()
