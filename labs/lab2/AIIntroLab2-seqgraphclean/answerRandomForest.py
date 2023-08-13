from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 30   # 树的数量
ratio_data = 0.6   # 采样的数据比例
ratio_feat = 0.8 # 采样的特征比例
hyperparams_forest = {"depth":40, "purity_bound":0.27, "gainfunc": negginiDA} # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
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
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
