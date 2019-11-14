import numpy as np

def corr2d(X, K):  # X输入，K为卷积核
    h, w = K.shape
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = np.array([[0, 1], [2, 3]])
print(corr2d(X, K))