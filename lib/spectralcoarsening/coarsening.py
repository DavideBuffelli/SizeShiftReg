import random
import numpy as np
import numpy.linalg as LA
import networkx as nx
from lib.spectralcoarsening import util
from lib.spectralcoarsening import measure
from lib.spectralcoarsening import laplacian
from sklearn.cluster import KMeans, SpectralClustering


def _multilevel_graph_coarsening(G, n):
    N = G.shape[0]
    Gc = G
    cur_size = N
    Q = np.eye(N)
    while cur_size > n:
        stop_flag = 0
        max_dist = 10000
        max_dist_a = -1
        max_dist_b = -1
        for i in range(cur_size):
            for j in range(i+1, cur_size):
                dist = measure.normalized_L1(Gc[i], Gc[j])
                if dist < max_dist:
                    max_dist = dist
                    max_dist_a = i
                    max_dist_b = j
                if dist < 0.001:
                    stop_flag = 1
                    break
            if stop_flag == 1:
                break
        if max_dist_a == -1:
            max_dist_a, max_dist_b = util.random_two_nodes(cur_size)
        cur_Q = util.merge_two_nodes(cur_size, max_dist_a, max_dist_b)
        Q = np.dot(Q, cur_Q)
        Gc = util.multiply_Q(Gc, cur_Q)
        cur_size = cur_size - 1
    idx = util.Q2idx(Q)
    return Gc, Q, idx

def regular_partition(N, n):
    if N%n == 0:
        block_size = N//n + 1
    else:
        block_size = N//(n-1) + 1
    idx = np.zeros(N, dtype=np.int32)
    for i in range(N):
        idx[i] = i//block_size
    return idx


def _spectral_graph_coarsening(G, n):
    N = G.shape[0]
    e1, v1, e2, v2 = laplacian.spectraLaplacian_two_end_n(G, n)
    min_dist = n+1

    for k in range(0, n):
        if e1[k] <= 1:
            if k+1 < n and e2[k+1] < 1:
                continue
            if k+1 < n and e2[k+1] >= 1:
                v_all = np.concatenate((v1[:, 0:(k+1)], v2[:, (k+1):n]), axis = 1)
            elif k == n-1:
                v_all = v1[:, 0:n]

            kmeans = KMeans(n_clusters=n).fit(v_all)
            idx = kmeans.labels_
            sumd = kmeans.inertia_
            Q = util.idx2Q(idx, n)
            Gc = util.multiply_Q(G, Q)
            ec, vc = laplacian.spectraLaplacian(Gc)
            dist = measure.eig_partial_dist_k_two_end_n(e1, e2, ec, k)
            if dist < min_dist:
                min_dist = dist
                idx_min = idx
                min_sumd = sumd
                Gc_min = Gc
                Q_min = Q
    return Gc_min, Q_min, idx_min


def spectral_graph_coarsening_lambda(G, n):
    N = G.shape[0]
    e1, v1, e2, v2 = laplacian.spectraLaplacian_two_end_n(G, n)
    min_dist = n+1

    for k in range(0, n):
        if e1[k] <= 1:
            if k+1 < n and e2[k+1] < 1:
                continue
            if k+1 < n and e2[k+1] >= 1:
                v_all = np.concatenate((v1[:, 0:(k+1)], v2[:, (k+1):n]), axis = 1)
            elif k == n-1:
                v_all = v1[:, 0:n]

            kmeans = KMeans(n_clusters=n).fit(v_all)
            idx = kmeans.labels_
            sumd = kmeans.inertia_
            Q = util.idx2Q(idx, n)
            Gc = util.multiply_Q(G, Q)
            ec, vc = laplacian.spectraLaplacian(Gc)
            dist = measure.eig_partial_dist_k_two_end_n(e1, e2, ec, k)
            if dist < min_dist:
                min_dist = dist
                idx_min = idx
                min_sumd = sumd
                Gc_min = Gc
                Q_min = Q
    return Gc_min, Q_min, idx_min

def spectral_clustering(G, n):
    N = G.shape[0]
    e1, v1 = laplacian.spectraLaplacian_top_n(G, n)
    v_all = v1[:, 0:n]
    kmeans = KMeans(n_clusters=n).fit(v_all)
    idx = kmeans.labels_
    sumd = kmeans.inertia_
    Q = util.idx2Q(idx, n)
    Gc = util.multiply_Q(G, Q)
    return Gc, Q, idx

def get_random_partition(N, n):
    for i in range(500):
        flag = True
        a = np.zeros(N, dtype=np.int64)
        cnt = np.zeros(n, dtype=np.int64)
        for j in range(N):
            a[j] = random.randint(0, n-1)
            cnt[a[j]] += 1
        for j in range(n):
            if cnt[j] == 0:
                flag = False
                break
        if flag == False:
            continue
        else:
            break
    return a


def get_nodes_for_cluster(cluster_labels):
    nodes_for_cluster = {}
    for node, cluster_label in enumerate(cluster_labels):
        if cluster_label not in nodes_for_cluster:
            nodes_for_cluster[cluster_label] = []
        nodes_for_cluster[cluster_label].append(node)
    return nodes_for_cluster


def get_coarsened_graph_from_clustering(cluster_labels, og_G):
    coarsened_graph = nx.Graph()
    nodes_for_cluster = get_nodes_for_cluster(cluster_labels) 

    # for when clustering leads to less nodes than what you asked for
    #missing = []
    #for x in range(n_clusters):
    #    if x not in nodes_for_cluster:
    #        #nodes_for_cluster[x] = [i for i in range(og_G.shape[0])]
    #        missing.append(x)
    #if len(missing) > 0:
    #    for m in missing[::-1]:
    #        for x in range(m+1, n_clusters):
    #            if x in nodes_for_cluster:
    #                nodes_for_cluster[x-1] = nodes_for_cluster.pop(x)

    for cluster in nodes_for_cluster:
        coarsened_graph.add_node(cluster, nodes_in_cluster=nodes_for_cluster[cluster])
    num_clusters = len(nodes_for_cluster)
    for cluster1 in range(num_clusters):
        for cluster2 in range(cluster1, num_clusters):
            for n1 in nodes_for_cluster[cluster1]:
                for n2 in nodes_for_cluster[cluster2]:
                    if og_G[n1,n2] == 1:
                        if not coarsened_graph.has_edge(cluster1, cluster2):
                            coarsened_graph.add_edge(cluster1, cluster2)
    return coarsened_graph, nodes_for_cluster


def spectral_clustering_coarsening(G, n):
    sc = SpectralClustering(n_clusters=n, affinity="precomputed")
    cluster_labels = sc.fit_predict(G)
    coarsened_graph, nodes_for_cluster = get_coarsened_graph_from_clustering(cluster_labels, G)
    return coarsened_graph, nodes_for_cluster


def kmeans_graph_coarsening(G, n):
    kmc = KMeans(n_clusters=n)
    cluster_labels = kmc.fit_predict(G)
    coarsened_graph, nodes_for_cluster = get_coarsened_graph_from_clustering(cluster_labels, G)
    return coarsened_graph, nodes_for_cluster


def multilevel_graph_coarsening(G, n):
    Gc, Q, cluster_labels = _multilevel_graph_coarsening(G, n)
    nodes_for_cluster = get_nodes_for_cluster(cluster_labels)
    coarsened_graph = nx.from_numpy_array(Gc)
    return coarsened_graph, nodes_for_cluster


def spectral_graph_coarsening(G, n):
    Gc, Q, cluster_labels = _spectral_graph_coarsening(G, n)
    nodes_for_cluster = get_nodes_for_cluster(cluster_labels)
    coarsened_graph = nx.from_numpy_array(Gc)
    return coarsened_graph, nodes_for_cluster
