from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

num_tree = 30   
ratio_data = 0.6   
ratio_feat = 0.8 
hyperparams_forest = {"depth":40, "purity_bound":0.27, "gainfunc": negginiDA} 


def buildtrees(X, Y):
    n, d = X.shape
    trees = []
    for i in range(num_tree):
        print("Begin build tree", i)
        # print(n, int(n * ratio_data), d, int(d * ratio_feat))
        sample_indices = np.random.choice(n, int(n * ratio_data), replace=False)
        feature_indices = np.random.choice(d, int(d * ratio_data), replace=False)
        
        # sample_indices = np.random.choice(n, int(n * ratio_data), replace=False)
        # feature_indices = np.random.choice(d, int(d * ratio_feat), replace=False)
        sample_X, sample_Y = X[sample_indices], Y[sample_indices]
        # sample_X = sample_X[:, feature_indices]
        root = buildTree(sample_X, sample_Y, feature_indices.tolist(), **hyperparams_forest)
        trees.append(root)
        print("Finish build tree", i)
    return trees

def infertrees(trees, X):
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]


from modelTree import *
from numpy.random import rand
from answerRandomForest import buildtrees, infertrees
from answerTree import *
from modelTree import discretize, trn_X, trn_Y, val_X, val_Y
from util import setseed

setseed(0) # 固定随机数种子以提高可复现性

save_path = "model/mymodel.npy"

if __name__ == "__main__":
    hyperparams["gainfunc"] = eval(hyperparams["gainfunc"])
    roots = buildtrees(trn_X, trn_Y)
    with open(save_path, "wb") as f:
        pickle.dump(roots, f)
    pred = np.array([infertrees(roots, val_X[i]) for i in range(val_X.shape[0])])
    print("valid acc", np.average(pred==val_Y))