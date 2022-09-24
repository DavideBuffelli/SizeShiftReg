import random
import numpy as np
import numpy.linalg as LA
from scipy import sparse

def erdos(n, p):
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            tmp = random.random()
            if tmp <= p:
                G[i, j] = 1
                G[j, i] = 1
    sum_G = np.sum(G, 1)
    for i in range(n):
        if sum_G[i] == 0:
            j = random.randint(0, n-1)
            G[i, j] = 1
            G[j, i] = 1
    return G


def sbm_pq(n, k, p, q):
    G = np.zeros((n, n))
    block_size = n//k
    for i in range(n):
        for j in range(n):
            tmp = random.random()
            if i//block_size == j // block_size:
                if tmp <= p:
                    G[i, j] = 1
                    G[j, i] = 1
            else:
                if tmp <= q:
                    G[i, j] = 1
                    G[j, i] = 1
    sum_G = np.sum(G, 1)
    for i in range(n):
        if sum_G[i] == 0:
            j = random.randint(0, n-1)
            G[i, j] = 1
            G[j, i] = 1
    return G

def sbm_qp(n, k, p, q):
    G = sbm_pq(n, k, q, p)
    return G

def sbm_pq_mixed(n, k, p, q):
    G = np.zeros((n, n))
    block_size = n//k
    B = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            tmp = random.random()
            if tmp < 1/2:
                B[i, j] = p
                B[j, i] = p
            else:
                B[i, j] = q
                B[j, i] = q
    for i in range(n):
        for j in range(n):
            tmp = random.random()
            if tmp <= B[i//block_size, j // block_size]:
                G[i, j] = 1
                G[j, i] = 1

    sum_G = np.sum(G, 1)
    for i in range(n):
        if sum_G[i] == 0:
            j = random.randint(0, n-1)
            G[i, j] = 1
            G[j, i] = 1
    return G
