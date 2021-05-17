import glob
import os
import pickle
import networkx as nx
import numpy as np

from classifier import Classifier
from utils import *
from support.Text2vec import sentence_emb
from support.Node2vec import node2vec_emb


def model_eval(original_G_path, content_emb_path, model_iter):
    build_graph(bg=False)
    if not os.path.exists(os.path.abspath('data/embedding/sentence_embedding/sentence_embedding.pkl')):
        sentence_emb.edge_content_emb()
        sentence_emb.node_content_emb()
    comp_helper(original_G_path=original_G_path, content_emb_path=content_emb_path, classifier='MLP',
                model_iter=model_iter)


def process_pred(original_G_path, structural_emb_path, content_emb_path, classifier, model_iter):
    binding = []
    for i in range(model_iter):
        # read from file and re-establish a copy of the original network
        full_G = nx.read_gml(original_G_path)

        # read the training_G from file and obtain a dict of {node: node_emb}
        training_G, structural_emb_dict = load_graph(original_G_path, structural_emb_path=structural_emb_path)
        # load content embedding information from that file
        content_emb_dict = pickle.load(open(content_emb_path, 'rb'))

        X_train, y_train, X_test_negatives, y_test_negatives, all_selected_indices, index2pair_dict \
            = load_training_data(full_G, full_G, structural_emb_dict, content_emb_dict, 0)

        clf = Classifier(X_train, y_train, None, None, classifier)
        clf.train()
        if i == model_iter - 1:
            clf.predict(full_G=full_G, all_selected_indices=all_selected_indices,
                        index2pair_dict=index2pair_dict, binding=binding, last_iter=True,
                        emb_name=str(structural_emb_path).rsplit('/', 1)[1].split('.')[0])
        else:
            clf.predict(full_G=full_G, all_selected_indices=all_selected_indices,
                        index2pair_dict=index2pair_dict, binding=binding, last_iter=False,
                        emb_name=str(structural_emb_path).rsplit('/', 1)[1].split('.')[0])


def model_pred(original_G_path, content_emb_path, model_iter):
    generate_embedding_pred(bg=False)
    process_pred(original_G_path=original_G_path,
                 structural_emb_path=os.path.abspath('data/embedding/prediction/IMSP.csv'),
                 content_emb_path=content_emb_path, classifier='MLP', model_iter=model_iter)


def model_pred_alt(left_G_path, content_emb_path, model_iter):
    generate_embedding_pred_alt(bg=False, path=left_G_path)
    process_pred(original_G_path=left_G_path,
                 structural_emb_path=os.path.abspath('data/embedding/prediction/IMSP.csv'),
                 content_emb_path=content_emb_path, classifier='MLP', model_iter=model_iter)


def generate_embedding_pred(bg):
    # build network, ensure network exist before embedding
    build_graph(bg)
    if not os.path.exists(os.path.abspath('data/embedding/sentence_embedding/sentence_embedding.pkl')):
        # content embedding
        sentence_emb.edge_content_emb()
        sentence_emb.node_content_emb()
    if not os.path.exists(os.path.abspath('data/embedding/prediction/IMSP.csv')):
        node2vec_emb.node_structure_emb('pred', os.path.abspath('data/classifier/original_G.txt'),
                                        'weighted', dim=128, walk_len=10, num_walks=100, p=1, q=0.5)


def generate_embedding_pred_alt(bg, path):
    # build network, ensure network exist before embedding
    build_graph(bg)
    if not os.path.exists(os.path.abspath('data/embedding/sentence_embedding/sentence_embedding.pkl')):
        # content embedding
        sentence_emb.edge_content_emb()
        sentence_emb.node_content_emb()
    node2vec_emb.node_structure_emb('pred', path,
                                    'weighted', dim=128, walk_len=10, num_walks=100, p=1, q=0.5)


def process_eval_IMSP(original_G_path, structural_emb_path, content_emb_path, classifier, i, fold):
    # read from file and re-establish a copy of the original network
    full_G = nx.read_gml(original_G_path)

    training_G_path, num_of_test_edges = establish_training_G(G=full_G, run=(i + 1), fold=(fold + 1), build_G=False)

    # read the training_G from file and obtain a dict of {node: node_emb}
    training_G, structural_emb_dict = load_graph(training_G_path, structural_emb_path=structural_emb_path)

    # load content embedding information from that file
    content_emb_dict = pickle.load(open(content_emb_path, 'rb'))

    full_G = nx.read_gml(original_G_path)

    X_train, y_train, X_test_negatives, y_test_negatives, all_selected_indices, index2pair_dict \
        = load_training_data(full_G, training_G, structural_emb_dict, content_emb_dict, num_of_test_edges)

    # test set is the set difference between edges in the original network and the new network
    full_G = nx.read_gml(original_G_path)
    # fill in the test set
    X_test, y_test = load_test_data(full_G,
                                    list(set(full_G.edges()) - set(training_G.edges())),
                                    structural_emb_dict, content_emb_dict,
                                    X_test_negatives, y_test_negatives)

    # get the prediction accuracy on evaluation set
    # for model in classification_models:
    clf = Classifier(X_train, y_train, X_test, y_test, classifier)
    clf.train()
    accuracy, report, macro_roc_auc_ovo, weighted_roc_auc_ovo = clf.test_model()
    return accuracy, report, macro_roc_auc_ovo, weighted_roc_auc_ovo


def comp_helper(original_G_path, content_emb_path, classifier, model_iter):
    print("In performing comparison")
    perf = {}
    for i in range(model_iter):
        print('-------------------- Iteration ' + str(i + 1) + ' / ' + str(model_iter) + ' --------------------')
        for fold in range(5):
            print('-------------------- Fold ' + str(fold + 1) + ' / ' + str(5) + ' --------------------')
            temp_G = nx.read_gml(original_G_path)
            establish_training_G(temp_G, (i + 1), (fold + 1), build_G=True)
            print('training graph built')
            generate_embedding_eval()
            structural_emb_path = glob.glob(os.path.abspath('data/embedding/evaluation/*.csv'))
            for emb in range(len(structural_emb_path)):
                emb_name = str(structural_emb_path[emb]).rsplit('/', 1)[1].split('.')[0]
                if 'IMSP' in str(structural_emb_path[emb]):
                    print('\nTesting our model IMSP')
                    acc, report, macro_roc_auc_ovo, weighted_roc_auc_ovo = \
                        process_eval_IMSP(original_G_path=original_G_path,
                                          structural_emb_path=structural_emb_path[emb],
                                          content_emb_path=content_emb_path,
                                          classifier=classifier,
                                          i=i,
                                          fold=fold)

                if i == 0 and fold == 0:
                    perf[emb_name] = {}
                    perf[emb_name]['PPI_f1'] = []
                    perf[emb_name]['PPI_precision'] = []
                    perf[emb_name]['PPI_recall'] = []
                    perf[emb_name]['infection_f1'] = []
                    perf[emb_name]['infection_precision'] = []
                    perf[emb_name]['infection_recall'] = []
                    perf[emb_name]['no_interaction_precision'] = []
                    perf[emb_name]['no_interaction_recall'] = []
                    perf[emb_name]['no_interaction_f1'] = []
                    perf[emb_name]['AUC_score_macro'] = []
                    perf[emb_name]['AUC_score_weighted'] = []
                    perf[emb_name]['accuracy'] = []
                    perf[emb_name]['weighted_f1'] = []
                    perf[emb_name]['weighted_precision'] = []
                    perf[emb_name]['weighted_recall'] = []

                append_res(PPI_f1=perf[emb_name]['PPI_f1'],
                           PPI_precision=perf[emb_name]['PPI_precision'],
                           PPI_recall=perf[emb_name]['PPI_recall'],
                           No_int_f1=perf[emb_name]['no_interaction_f1'],
                           No_int_precision=perf[emb_name]['no_interaction_precision'],
                           No_int_recall=perf[emb_name]['no_interaction_recall'],
                           AUC_score_macro=perf[emb_name]['AUC_score_macro'],
                           AUC_score_weighted=perf[emb_name]['AUC_score_weighted'],
                           acc=acc,
                           accuracy=perf[emb_name]['accuracy'],
                           infection_f1=perf[emb_name]['infection_f1'],
                           infection_precision=perf[emb_name]['infection_precision'],
                           infection_recall=perf[emb_name]['infection_recall'],
                           macro_roc_auc_ovo=macro_roc_auc_ovo,
                           report=report,
                           weighted_f1=perf[emb_name]['weighted_f1'],
                           weighted_precision=perf[emb_name]['weighted_precision'],
                           weighted_recall=perf[emb_name]['weighted_recall'],
                           weighted_roc_auc_ovo=weighted_roc_auc_ovo)

    # write overall performance
    with open(os.path.abspath('data/evaluation/performance_summary.csv'), 'w') as file:
        file.write(
            'Embedding Model,No Interaction Precision,SD,No Interaction Recall,SD,No Interaction F1-score,SD,'
            'Infection Precision,SD,Infection Recall,SD,Infection F1-score,SD,PPI Precision,SD,PPI Recall,SD,'
            'PPI F1-score,SD,Accuracy,SD,Weighted Precision,SD,Weighted Recall,SD,Weighted F1-score,SD,'
            'AUC macro,SD,AUC weighted,SD\n'
        )
        structural_emb_path = glob.glob(os.path.abspath('data/embedding/evaluation/*.csv'))
        for emb in range(len(structural_emb_path)):
            emb_name = str(structural_emb_path[emb]).rsplit('/', 1)[1].split('.')[0]
            # write model performance to file after multiple iterations are complete
            to_write = write_res(emb_name=emb_name,
                                 PPI_f1=perf[emb_name]['PPI_f1'],
                                 PPI_precision=perf[emb_name]['PPI_precision'],
                                 PPI_recall=perf[emb_name]['PPI_recall'],
                                 no_int_f1=perf[emb_name]['no_interaction_f1'],
                                 no_int_precision=perf[emb_name]['no_interaction_precision'],
                                 no_int_recall=perf[emb_name]['no_interaction_recall'],
                                 AUC_score_macro=perf[emb_name]['AUC_score_macro'],
                                 AUC_score_weighted=perf[emb_name]['AUC_score_weighted'],
                                 accuracy=perf[emb_name]['accuracy'],
                                 infection_f1=perf[emb_name]['infection_f1'],
                                 infection_precision=perf[emb_name]['infection_precision'],
                                 infection_recall=perf[emb_name]['infection_recall'],
                                 weighted_f1=perf[emb_name]['weighted_f1'],
                                 weighted_precision=perf[emb_name]['weighted_precision'],
                                 weighted_recall=perf[emb_name]['weighted_recall'])
            file.write(to_write)

    # write performance details
    with open(os.path.abspath('data/evaluation/performance_details.csv'), 'w') as detail_file:
        detail_file.write(
            'Embedding Model,No interaction Precision,No interaction Recall,No interaction F1-score,'
            'Infection Precision,Infection Recall,Infection F1-score,PPI Precision,PPI Recall,'
            'PPI F1-score,Accuracy,Weighted Precision,Weighted Recall,Weighted F1-score,AUC macro,AUC weighted\n'
        )
        for emb in range(len(structural_emb_path)):
            emb_name = str(structural_emb_path[emb]).rsplit('/', 1)[1].split('.')[0]
            # write model performance to file after multiple iterations are complete
            to_write = write_res_details(emb_name=emb_name,
                                         PPI_f1=perf[emb_name]['PPI_f1'],
                                         PPI_precision=perf[emb_name]['PPI_precision'],
                                         PPI_recall=perf[emb_name]['PPI_recall'],
                                         no_int_f1=perf[emb_name]['no_interaction_f1'],
                                         no_int_precision=perf[emb_name]['no_interaction_precision'],
                                         no_int_recall=perf[emb_name]['no_interaction_recall'],
                                         AUC_score_macro=perf[emb_name]['AUC_score_macro'],
                                         AUC_score_weighted=perf[emb_name]['AUC_score_weighted'],
                                         accuracy=perf[emb_name]['accuracy'],
                                         infection_f1=perf[emb_name]['infection_f1'],
                                         infection_precision=perf[emb_name]['infection_precision'],
                                         infection_recall=perf[emb_name]['infection_recall'],
                                         weighted_f1=perf[emb_name]['weighted_f1'],
                                         weighted_precision=perf[emb_name]['weighted_precision'],
                                         weighted_recall=perf[emb_name]['weighted_recall']
                                         )
            detail_file.write(to_write)


def generate_embedding_eval():
    print("network embedding for performance evaluation")
    # IMSP
    node2vec_emb.node_structure_emb('eval', os.path.abspath('data/classifier/training_G.txt'), 'weighted',
                                    dim=128, walk_len=10, num_walks=100, p=1, q=0.5)

    print("network embedding for performance evaluation finished")


def write_res(emb_name, PPI_f1, PPI_precision, PPI_recall, no_int_f1, no_int_precision, no_int_recall, AUC_score_macro,
              AUC_score_weighted, accuracy, infection_f1,
              infection_precision, infection_recall, weighted_f1, weighted_precision, weighted_recall):
    to_write = emb_name + ',' + str(
        np.mean(no_int_precision)) + ',' + str(np.std(no_int_precision)) + ',' + str(
        np.mean(no_int_recall)) + ',' + str(np.std(no_int_recall)) + ',' + str(
        np.mean(no_int_f1)) + ',' + str(np.std(no_int_f1)) + ',' + str(
        np.mean(infection_precision)) + ',' + str(np.std(infection_precision)) + ',' + str(
        np.mean(infection_recall)) + ',' + str(np.std(infection_recall)) + ',' + str(
        np.mean(infection_f1)) + ',' + str(np.std(infection_f1)) + ',' + str(
        np.mean(PPI_precision)) + ',' + str(np.std(PPI_precision)) + ',' + str(
        np.mean(PPI_recall)) + ',' + str(np.std(PPI_recall)) + ',' + str(
        np.mean(PPI_f1)) + ',' + str(np.std(PPI_f1)) + ',' + str(
        np.mean(accuracy)) + ',' + str(np.std(accuracy)) + ',' + str(
        np.mean(weighted_precision)) + ',' + str(np.std(weighted_precision)) + ',' + str(
        np.mean(weighted_recall)) + ',' + str(np.std(weighted_recall)) + ',' + str(
        np.mean(weighted_f1)) + ',' + str(np.std(weighted_f1)) + ',' + str(
        np.mean(AUC_score_macro)) + ',' + str(np.std(AUC_score_macro)) + ',' + str(
        np.mean(AUC_score_weighted)) + ',' + str(np.std(AUC_score_weighted)) + '\n'
    return to_write


def write_res_details(emb_name, PPI_f1, PPI_precision, PPI_recall, no_int_f1, no_int_precision, no_int_recall,
                      AUC_score_macro, AUC_score_weighted, accuracy,
                      infection_f1, infection_precision, infection_recall, weighted_f1, weighted_precision,
                      weighted_recall):
    to_write = ''
    for i in range(len(PPI_f1)):
        to_write = to_write + emb_name + ',' + str(
            no_int_precision[i]) + ',' + str(
            no_int_recall[i]) + ',' + str(
            no_int_f1[i]) + ',' + str(
            infection_precision[i]) + ',' + str(
            infection_recall[i]) + ',' + str(
            infection_f1[i]) + ',' + str(
            PPI_precision[i]) + ',' + str(
            PPI_recall[i]) + ',' + str(
            PPI_f1[i]) + ',' + str(
            accuracy[i]) + ',' + str(
            weighted_precision[i]) + ',' + str(
            weighted_recall[i]) + ',' + str(
            weighted_f1[i]) + ',' + str(
            AUC_score_macro[i]) + ',' + str(
            AUC_score_weighted[i]) + ',' + '\n'

    return to_write


def append_res(PPI_f1, PPI_precision, PPI_recall, No_int_f1, No_int_precision, No_int_recall, AUC_score_macro,
               AUC_score_weighted, acc, accuracy, infection_f1,
               infection_precision, infection_recall, macro_roc_auc_ovo, report, weighted_f1, weighted_precision,
               weighted_recall, weighted_roc_auc_ovo):
    No_int_precision.append(report['No interaction']['precision'])
    infection_precision.append(report['Infection']['precision'])
    PPI_precision.append(report['PPI']['precision'])
    No_int_recall.append(report['No interaction']['recall'])
    infection_recall.append(report['Infection']['recall'])
    PPI_recall.append(report['PPI']['recall'])
    No_int_f1.append(report['No interaction']['f1-score'])
    infection_f1.append(report['Infection']['f1-score'])
    PPI_f1.append(report['PPI']['f1-score'])
    accuracy.append(acc)
    weighted_precision.append(report['weighted avg']['precision'])
    weighted_recall.append(report['weighted avg']['recall'])
    weighted_f1.append(report['weighted avg']['f1-score'])
    AUC_score_macro.append(macro_roc_auc_ovo)
    AUC_score_weighted.append(weighted_roc_auc_ovo)
