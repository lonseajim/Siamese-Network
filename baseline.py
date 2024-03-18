import numpy as np


# 计算欧氏距离
def dist_Euclidean(vec_a, vec_b):
    return np.sqrt(np.sum(np.power((vec_a - vec_b), 2)))


# 计算曼哈顿距离
def dist_Manhattan(vec_a, vec_b):
    return np.sum(np.abs(vec_a - vec_b))


# 计算余弦相似度
def dis_cosine(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

