import math
import pandas as pd
import numpy as np
import argparse

SPLIT_NOT_THIS_SIDE = "*"
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
MAX_SCORE = 500
MIN_SCORE = -500
class XGBTree:
    T:int # num of trees
    d:int # depth
    splits:list() # list of splits
    weights:list() # weights
    
    def __init__(self, T, d, splits, weights):
        self.T = T
        self.d = d
        self.splits = splits
        self.weights = weights
    # whether assign sample to left
    def direct_to_left(self, x, split):
        feature_value = x[split[0]]
        if feature_value < split[2]-0.00001:
            return True
        else:
            return False
        
    # sigomid function
    def sigmoid(self, x):
        return 1.0/(1.0 + math.e ** -x)
        
    def cal_score(self, x):
        score = 0.0
        for i in range(T):
            pos = 0
            for j in range(d-1):
                split = self.splits[i][j][pos]
                if self.direct_to_left(x, split):
                    # to left
                    pos = pos * 2
                else:
                    # to right
                    pos = pos * 2 + 1
            score = score + self.weights[i][pos]
        score = max(score, MIN_SCORE)
        score = min(score, MAX_SCORE)
        score = self.sigmoid(score)
        return score
            
    def infer(self, X:np.ndarray):
        y_estimate = []
        for sample in X:
            score = self.cal_score(sample)
            label = POSITIVE_LABEL if score >= 0.5 else NEGATIVE_LABEL
            y_estimate.append(label)
        return y_estimate
        
# read model
def read_model(model_name):
    splits = []
    weights =[]
    with open(model_name) as file:
        T, d, m, log2scale = [int(x) for x in next(file).strip().split()]
        for i in range(T):
            splits_level = []
            for j in range(d-1):
                splits_level.append(next(file).strip().split(";"))
            weights_level = next(file).strip().split()
            splits.append(splits_level)
            weights.append(weights_level)
    return splits, weights, m, T, d, log2scale

# recovery shares
def recovery_from_shares(x, y, log2scale):
    xy = (int(x) + int(y)) % (2**64)
    res = (xy - 2**64) if (xy >= 2**63) else xy
    return res/(2**log2scale)

# normalize lable
def label_norm(y):
    y = [int(item) for item in y]
    positive_label = max(y)
    negative_label = min(y)
    y = [POSITIVE_LABEL if item == positive_label else NEGATIVE_LABEL for item in y]
    return y

# print tree
def print_tree(T, d, splits, weights):
    for i in range(T):
        print(i, "-th tree")
        for j in range(d-1):
            print(j,"-th level", splits[i][j])
        print("weights: ", weights[i])

# calculate F1 score
def cal_F1_score(y, y_estimate, verbose = False):
    FP = 0
    TP = 0
    FN = 0
    TN = 0
    for (y0, y1) in zip(y, y_estimate):
        if y0 == y1:
            if y0 == POSITIVE_LABEL:
                TP = TP + 1
            else:
                TN = TN + 1
        else:
            if y0 == POSITIVE_LABEL:
                FN = FN + 1
            else:
                FP = FP + 1
    F1_score = 2 * TP /(2 * TP + FP + FN)
    if verbose:
        print("TP=",TP)
        print("TN=",TN)
        print("FP=",FP)
        print("FN=",FN)
        print("F1=",F1_score)
    return F1_score



parser = argparse.ArgumentParser(description="XGB inference: avg F1 score over all k-fold test")
parser.add_argument("--test_path", type=str, help="test file path")
parser.add_argument("--model_path", type=str, help="model path")
parser.add_argument("--dataset", type=str, help="dataset name")
args = parser.parse_args()

res = []
# k_fold evaluation
for k_fold in range(5):
    test_file_name = args.test_path+args.dataset+"_test_" + str(k_fold) + ".csv"
    model_Alice_file = args.model_path+args.dataset+"_model_Alice_" + str(k_fold) + ".txt"
    model_Bob_file = args.model_path+args.dataset+"_model_Bob_" + str(k_fold) + ".txt"
    print(test_file_name)
    print(model_Alice_file)
    print(model_Bob_file)

    splits_alice,weights_alice, m_alice, T, d, log2scale = read_model(model_Alice_file)
    splits_bob, weights_bob, m_bob, _, _, _ = read_model(model_Bob_file)
    print("Alice's #feature = ", m_alice)
    print("Bob's #feature = ", m_bob)
    splits = []
    weights = []
    # for each tree
    for i in range(T):
        splits_tree = []
        # for each level
        for j in range(d-1):
            splits_level = []
            for split_alice, split_bob in zip(splits_alice[i][j], splits_bob[i][j]):
                current_split = split_bob if SPLIT_NOT_THIS_SIDE in split_alice else split_alice
                ind_feature, split_lower, split_upper = \
                    [float(x) for x in current_split.replace("[", "").replace("]", "").replace(")", "").split(",")]
                ind_feature = (int(ind_feature) + m_alice) if SPLIT_NOT_THIS_SIDE in split_alice else int(ind_feature)
                splits_level.append((ind_feature, split_lower, split_upper))
            splits_tree.append(splits_level)    
        splits.append(splits_tree)
        # read weights
        weights_level = []
        for wa, wb in zip(weights_alice[i], weights_bob[i]):
            weights_level.append(recovery_from_shares(wa, wb,log2scale))
        weights.append(weights_level)

    print_tree(T,d, splits,weights)
        
    df = pd.read_csv(test_file_name, header=None, index_col=0)

    # extrat x and y
    y = df.iloc[:,0].to_list()
    df_X = df.iloc[:,1:]
    X = df_X.to_numpy()
    y = label_norm(y)

    # infer
    xgb_tree = XGBTree(T, d, splits, weights)
    y_estimate = xgb_tree.infer(X)
    F1_score = cal_F1_score(y, y_estimate, verbose=True)
    res.append(F1_score)
print(res)
print ("avg F1 = ", sum(res)/len(res))


    
    
    
