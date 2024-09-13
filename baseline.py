import numpy as np


# Euclidean
def dist_Euclidean(vec_a, vec_b):
    return np.sqrt(np.sum(np.power((vec_a - vec_b), 2)))


# Manhattan
def dist_Manhattan(vec_a, vec_b):
    return np.sum(np.abs(vec_a - vec_b))


# cosine
def dist_cosine(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
