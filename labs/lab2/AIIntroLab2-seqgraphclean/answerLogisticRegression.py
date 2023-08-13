import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 0.048 # 学习率
# wd = 1e-5  # l2正则化项系数
eps = 1e-6  # 防止除0错误


def predict(X, weight, bias):
    """
    使用输入的weight和bias预测样本X是否为数字0
    @param X: n*d 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: d*1
    @param bias: 1*1
    @return: wx+b
    """
    ans = np.dot(X, weight) + bias
    # print("ans shape", ans.shape)
    return ans

def sigmoid(x):
    x_ravel = x.ravel()
    y = []
    for i in range(len(x_ravel)):
        xval = x_ravel[i]
        if xval > 0:
            tmp = 1 / (1 + np.exp(-xval))
        else:
            tmp = np.exp(xval) / (1 + np.exp(xval))
        y.append(tmp)
    y = np.array(y)
    return y

def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: n*d 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: d*1
    @param bias: 1*1
    @param Y: n 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: n 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: 1*1 由交叉熵损失函数计算得到
        weight: d*1 更新后的weight参数
        bias: 1*1 更新后的bias参数
    """
    n, d = X.shape
    # print(weight.shape)
    logistic = sigmoid(predict(X, weight, bias) * Y)
    ones = np.ones((n, ))
    # print(tmp.shape, weight.shape)
    weight += lr * np.dot(X.T, (ones - logistic) * Y) / n
    bias += lr * np.sum((ones - logistic) * Y) / n
    logistic = sigmoid(predict(X, weight, bias) * Y)
    loss = -np.sum(np.log(logistic + eps)) / n
    prob = sigmoid(predict(X, weight, bias))
    haty = []
    for i in range(n):
        if prob[i] > 0.5:
            haty.append(1)
        else:
            haty.append(-1)
    haty = np.array(haty)
    return haty, loss, weight, bias
