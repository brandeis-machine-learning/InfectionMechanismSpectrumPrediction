import numpy as np
from network.network_data import known_negatives as known_neg
import utils as utils


def random_sampling_helper(training_G, X, y, positives, negatives, num_of_training_edges,
                           num_of_test_edges, index2pair_dict, all_selected_indices):
    X_train = np.empty([4 * num_of_training_edges, X.shape[1]])
    y_train = np.empty([4 * num_of_training_edges, 1])

    # positive training sample
    count = 0
    for idx in positives:
        pair = index2pair_dict[idx]
        if (str(pair[0]), str(pair[1])) in training_G.edges():
            X_train[count] = X[idx]
            y_train[count] = y[idx]
            count = count + 1

    # negative samples
    X_test_negatives = np.empty([2 * num_of_test_edges, X.shape[1]])
    y_test_negatives = np.empty([2 * num_of_test_edges, 1])

    known_negatives = get_known_negatives(training_G, X)
    unknown_negatives = np.setdiff1d(negatives, known_negatives)

    if num_of_test_edges == 0:
        known_negatives_testing = np.random.choice(known_negatives,
                                                   size=0,
                                                   replace=False)
    else:
        known_negatives_testing = np.random.choice(known_negatives,
                                                   size=int(len(known_negatives) * 0.2),
                                                   replace=False)
    known_negatives_training = np.setdiff1d(known_negatives, known_negatives_testing)

    selected_unknown_negatives = np.random.choice(unknown_negatives,
                                                  size=2 * (num_of_training_edges + num_of_test_edges) - len(
                                                      known_negatives),
                                                  replace=False)

    if num_of_test_edges == 0:
        unknown_negatives_training = np.random.choice(selected_unknown_negatives,
                                                      size=len(selected_unknown_negatives),
                                                      replace=False)

    else:
        unknown_negatives_training = np.random.choice(selected_unknown_negatives,
                                                      size=2 * num_of_training_edges - (len(known_negatives) - int(
                                                          len(known_negatives) * 0.2)),
                                                      replace=False)
    unknown_negatives_testing = np.setdiff1d(selected_unknown_negatives, unknown_negatives_training)

    selected_indices_for_training = np.concatenate((known_negatives_training, unknown_negatives_training))

    selected_indices_for_testing = np.concatenate((known_negatives_testing, unknown_negatives_testing))

    for idx in selected_indices_for_training:
        X_train[count] = X[idx]
        y_train[count] = y[idx]
        all_selected_indices.append(idx)
        count = count + 1

    count = 0
    if len(selected_indices_for_testing) > 0:
        for idx in selected_indices_for_testing:
            X_test_negatives[count] = X[idx]
            y_test_negatives[count] = y[idx]
            all_selected_indices.append(idx)
            count = count + 1

    return X_train, y_train, X_test_negatives, y_test_negatives


def get_known_negatives(training_G, X):
    known_negatives = []

    mers = "Middle East respiratory syndrome-related coronavirus"
    sars = "Severe acute respiratory syndrome-related coronavirus"
    sars2 = "Severe acute respiratory syndrome coronavirus 2"
    nl63 = "Human coronavirus NL63"
    ACE2s = [sars, sars2, nl63]

    index2pair_dict = utils.get_index2pair_dict(len(X), training_G)
    for key in index2pair_dict:
        val = index2pair_dict[key]
        src_node = str(val[0])
        dst_node = str(val[1])
        if (training_G.nodes[src_node]['type'] == 'Spike'
            and training_G.nodes[src_node]['host'] == mers
            and training_G.nodes[dst_node]['type'] == 'ACE2') or \
                (training_G.nodes[src_node]['type'] == 'Spike'
                 and training_G.nodes[src_node]['host'] in ACE2s
                 and training_G.nodes[dst_node]['type'] == 'DPP4') or \
                (training_G.nodes[dst_node]['type'] == 'Spike'
                 and training_G.nodes[dst_node]['host'] == mers
                 and training_G.nodes[src_node]['type'] == 'ACE2') or \
                (training_G.nodes[dst_node]['type'] == 'Spike'
                 and training_G.nodes[dst_node]['host'] in ACE2s
                 and training_G.nodes[src_node]['type'] == 'DPP4'):
            known_negatives.append(key)
        elif (training_G.nodes[src_node]['disp'], training_G.nodes[dst_node]['disp']) in known_neg:
            known_negatives.append(key)
    return known_negatives


def random_sampling(training_G, X, y, y_copy, num_of_training_edges, num_of_test_edges, index2pair_dict):
    all_selected_indices = []
    negative_idx_lst = []
    positive_idx_lst = []
    for idx in range(0, len(X)):
        if y[idx] == 0:
            negative_idx_lst.append(idx)
        elif y_copy[idx] >= 1:
            positive_idx_lst.append(idx)
        if y[idx] >= 1:
            all_selected_indices.append(idx)

    # shrink the larger negative_X to generate evaluation set and test set
    X_train, y_train, X_test_negatives, y_test_negatives = random_sampling_helper(training_G, X, y,
                                                                                  positive_idx_lst,
                                                                                  negative_idx_lst,
                                                                                  num_of_training_edges,
                                                                                  num_of_test_edges,
                                                                                  index2pair_dict,
                                                                                  all_selected_indices)

    return X_train, y_train, X_test_negatives, y_test_negatives, all_selected_indices
