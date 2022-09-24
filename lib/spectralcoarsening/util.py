import random
import numpy as np
import numpy.linalg as LA
import networkx as nx

from sklearn.cluster import KMeans


def multiply_Q(G, Q):
    Gc = np.dot(np.dot(np.transpose(Q), G), Q)
    return Gc

def multiply_Q_lift(Gc, Q):
    G = np.dot(np.dot(Q, Gc), np.transpose(Gc))
    return G

def idx2Q(idx, n):
    N = idx.shape[0]
    Q = np.zeros((N, n))
    for i in range(N):
        Q[i, idx[i]] = 1
    return Q

def Q2idx(Q):
    N = Q.shape[0]
    n = Q.shape[1]
    idx = np.zeros(N, dtype=np.int32)
    for i in range(N):
        for j in range(n):
            if Q[i, j] > 0:
                idx[i] = j
    return idx

def random_two_nodes(n):
    perm = np.random.permutation(n)
    return perm[0], perm[1]

def merge_two_nodes(n, a, b):

    assert a != b and a < n and b < n
    Q = np.zeros((n, n-1))
    cur = 0
    for i in range(n):
        if i == a or i == b:
            Q[i, n-2] = 1
        else:
            Q[i, cur] = 1
            cur = cur + 1
    return Q

def lift_Q(Q):
    N = Q.shape[0]
    n = Q.shape[1]
    idx = np.zeros(N, dtype=np.int16)
    for i in range(N):
        for j in range(n):
            if Q[i, j] == 1:
                idx[i] = j
    d = np.zeros((n, 1))
    for i in range(N):
        d[idx[i]] = d[idx[i]] + 1

    Q2 = np.zeros((N, n))
    for i in range(N):
        Q2[i, idx[i]] = 1/d[idx[i]]
    return Q2
