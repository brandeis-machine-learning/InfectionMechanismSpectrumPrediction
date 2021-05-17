"""import modules"""

from process import *
from utils import build_graph_alt
import cross_val as cv
import argparse


def main():
    bg = args.build_G
    evaluate = args.eval
    eval_iter = args.eval_iter
    leave_one_out = args.leave_one_out

    print("args: ", args)

    # --------------------------------------------------------
    # --------------------Configurations----------------------
    # --------------------------------------------------------

    # path for content embedding
    content_emb_path = os.path.abspath('data/embedding/sentence_embedding/sentence_embedding.pkl')

    # path for constructed network
    original_G_path = os.path.abspath('data/classifier/original_G.txt')

    # --------------------------------------------------------
    # ------------------------Execution-----------------------
    # --------------------------------------------------------
    # only construct network
    if bg:
        build_graph(bg=bg)
        return

    # evaluate model performance
    if evaluate:
        # get stratified CV results
        if os.path.exists(os.path.abspath('./data/evaluation/cross_validation/run-1-fold-1-X.csv')):
            print("CV splits found...")
        else:
            cv.stratified_k_fold(eval_iter)
        model_eval(original_G_path=original_G_path, content_emb_path=content_emb_path, model_iter=eval_iter)

    # perform prediction using IMSP
    else:
        # full graph prediction
        if not leave_one_out:
            model_pred(original_G_path=original_G_path, content_emb_path=content_emb_path, model_iter=1)
        # leave-out-out infection prediction for evaluation purpose
        else:
            ext_names = build_graph_alt(graph_path=original_G_path)
            print('hosts that can be left out are: ', ext_names)
            # specify which to leave-out
            print("left-out host: ", ext_names[1])
            model_pred_alt(left_G_path=os.path.abspath('data/classifier/original_G_' + ext_names[1] + '.txt'),
                           content_emb_path=content_emb_path, model_iter=1)

    print("\n-----------------END--------------------")


parser = argparse.ArgumentParser()

parser.add_argument("--eval", type=bool,
                    help="if set to True, the model evaluates the link prediction performance and performs comparison; " +
                         "if set to False, the model makes prediction. Default: False",
                    default=False)

parser.add_argument("--eval_iter", type=int, help="the number of runs while evaluating the performance. " +
                                                  "Default: 30",
                    default=30)

parser.add_argument("--leave_one_out", type=bool,
                    help="leave-out-out for infections prediction using IMSP. Default: false", default=False)

parser.add_argument('--build_G', type=bool,
                    help="if set to True, the model will stop once the network is built. Default: False",
                    default=False)

args = parser.parse_args()

main()

# if __name__ == '__main__':
#     main()
