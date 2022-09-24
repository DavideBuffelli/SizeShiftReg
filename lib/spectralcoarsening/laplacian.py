import random
import numpy as np
import numpy.linalg as LA
from scipy.sparse.linalg import eigs

from numpy.linalg import eig

#def Laplacian(G):


def normalizeLaplacian(G):
    n = G.shape[0]
    d = np.sum(G, axis = 0)
    d_inv_sqrt = 1/np.sqrt(d)

    # added to handle when there are isolated nodes
    d_inv_sqrt = np.nan_to_num(d_inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)

    # if d_inv_sqrt.shape[0] == 1:
    #     d_inv_sqrt = d_inv_sqrt[0]
    #     d_inv_sqrt_tmp = np.zeros((n, n))
    #     for i in range(n):
    #         d_inv_sqrt_tmp[i, i] = d_inv_sqrt[0, i]
    #     d_inv_sqrt = d_inv_sqrt_tmp
    # else:
    d_inv_sqrt = np.diag(d_inv_sqrt)
    return np.eye(n) - np.dot(np.dot(d_inv_sqrt, G), d_inv_sqrt)


def spectraLaplacian(G):
    L = normalizeLaplacian(G)
    L = (L + np.transpose(L)) / 2
    e, v = LA.eig(L)
    e = np.real(e)
    v = np.real(v)
    e_tmp = -e
    idx = e_tmp.argsort()[::-1]
    e = e[idx]
    v = v[:,idx]
    return e, v

def spectraLaplacian_two_end_n(G, n):
    N = G.shape[0]
    assert n <= N
    L = normalizeLaplacian(G)
    L = (L + np.transpose(L)) / 2
    e, v = eig(L)
    # e1, v1 = eigs(L, k = n, which = 'SM')
    # e1 = np.real(e1)
    # v1 = np.real(v1)
    # e1_tmp = -e1
    # idx = e1_tmp.argsort()[::-1]
    # e1 = e1[idx]
    # v1 = v1[:,idx]
    # e2, v2 = eigs(L, k = n, which = 'LM')
    # e2 = np.real(e2)
    # v2 = np.real(v2)
    # e2_tmp = -e2
    # idx = e2_tmp.argsort()[::-1]
    # e2 = e2[idx]
    # v2 = v2[:,idx]

    e = np.real(e)
    v = np.real(v)
    e_tmp = -e
    idx = e_tmp.argsort()[::-1]
    e = e[idx]
    v = v[:,idx]
    e1 = e[0:n]
    v1 = v[:, 0:n]
    e2 = e[N-n:N]
    v2 = v[:, N-n:N]
    return e1, v1, e2, v2

def spectraLaplacian_top_n(G, n):
    assert n <= G.shape[0]

    L = normalizeLaplacian(G)
    L = (L + np.transpose(L)) / 2

    e1, v1 = eig(L)
    # e1, v1 = eigs(L, k = n, which = 'SM')
    e1 = np.real(e1)
    v1 = np.real(v1)
    e1_tmp = -e1
    idx = e1_tmp.argsort()[::-1]
    e1 = e1[idx]
    e1 = e1[0:n]
    v1 = v1[:,idx]
    v1 = v1[:,0:n]
    return e1, v1
