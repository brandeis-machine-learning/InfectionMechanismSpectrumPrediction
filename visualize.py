import networkx as nx
import os
import csv
import json


def visualize_in_cytoscape():
    full_G = nx.read_gml(os.path.abspath('./data/classifier/original_G.txt'))
    with open(os.path.abspath('./data/prediction/prediction_IMSP_interacts.csv'), 'r') as int_pred:
        PPI_csv = csv.reader(int_pred, delimiter=',')
        for row in PPI_csv:
            if row[7] == 'likely':
                if full_G.has_edge(str(row[0]), str(row[1])):
                    print(row[0], row[1])
                full_G.add_edge(str(row[0]), str(row[1]), etype='predicted',
                                relation='interacts',
                                probability_estimate=row[4], connection=row[5])
        int_pred.close()

    with open(os.path.abspath('./data/prediction/prediction_IMSP_infects.csv'), 'r') as inf_pred:
        inf_csv = csv.reader(inf_pred, delimiter=',')
        for row in inf_csv:
            if row[6] == 'likely':
                if full_G.has_edge(str(row[0]), str(row[1])):
                    print(row[0], row[1])
                full_G.add_edge(str(row[0]), str(row[1]), etype='predicted',
                                relation='infects',
                                probability_estimate=row[4], connection=row[5])
        inf_pred.close()

    json_cyto = nx.cytoscape_data(full_G)
    with open('data/cytoscape/cytoscape_with_prediction.json', 'w') as json_file:
        json.dump(json_cyto, json_file)
