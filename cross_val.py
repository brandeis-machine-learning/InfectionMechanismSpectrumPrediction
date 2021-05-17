import networkx as nx
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import copy


def save_file(X_test, y_test, run, fold):
    """
    save to files
    @param X_test: X-fold
    @param y_test: y-fold
    @param run: run #
    @param fold: fold #
    @return:
    """
    with open(os.path.abspath('./data/evaluation/cross_validation/run-' + str(run) + '-fold-' + str(fold) + '-X.csv'),
              'w') as X:
        for edge in X_test:
            # print(edge)
            X.write(str(edge[0]) + ',' + str(edge[1]) + '\n')
        X.close()
    with open(os.path.abspath('./data/evaluation/cross_validation/run-' + str(run) + '-fold-' + str(fold) + '-y.csv'),
              'w') as y:
        for target in y_test:
            # print(target)
            y.write(str(target) + '\n')
        y.close()


# os.path.abspath('')
def stratified_k_fold(num_splits):
    """
    perform stratified k-fold splits on graph data
    @param num_splits:
    @return:
    """
    G = nx.read_gml(os.path.abspath('./data/classifier/original_G.txt'))
    X = []
    y = []
    for e in G.edges(data=True):
        edge_relation = e[2]['relation']
        if edge_relation.__contains__('similar'):
            y_to_add = 1
        if edge_relation.__contains__('infects'):
            y_to_add = 2
        elif edge_relation.__contains__('belongs'):
            y_to_add = 3
        elif edge_relation.__contains__('interacts'):
            y_to_add = 4
        X.append([int(e[0]), int(e[1])])
        y.append(y_to_add)

    X = np.array(X)
    y = np.array(y)

    run = 1
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    while run <= num_splits:
        temp_pred = 0
        curr_split = skf.split(X, y)
        X_folds = []
        y_folds = []
        for train_index, test_index in curr_split:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_folds.append(X_test)
            y_folds.append(y_test)
            # print(X_test)
            temp_G = nx.read_gml(os.path.abspath('./data/classifier/original_G.txt'))
            for edge in X_test:
                temp_G.remove_edge(str(edge[0]), str(edge[1]))
            if nx.is_connected(temp_G):
                temp_pred = temp_pred + 1
        # print(temp_pred)
        if temp_pred == 5:
            print('# of splits found: ', run)
            fold = 1
            for i in range(len(X_folds)):
                save_file(X_folds[i], y_folds[i], run, fold)
                fold = fold + 1
            run = run + 1
        else:
            print("this split cannot ensure full connectivity, re-doing...")


if __name__ == '__main__':
    stratified_k_fold(num_splits=30)
