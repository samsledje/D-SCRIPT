import numpy as np
import scipy.spatial.distance as spatial
from numpy import linalg as LA


def get_dim(edgelist):
    """Given an adjacency list for a graph, returns the number of nodes in
    the graph.

    :param edgelist: Graph adjacency list
    :type edgelist: list
    :return: Number of nodes in the graph
    :rtype: int
    """
    node_dict = {}
    node_count = 0
    for edge in edgelist:
        p, q = edge[:2]
        if p not in node_dict:
            node_dict[p] = True
            node_count += 1
        if q not in node_dict:
            node_dict[q] = True
            node_count += 1
    return node_count


def densify(edgelist, dim=None, directed=False):
    """Given an adjacency list for the graph, computes the adjacency
    matrix.

    :param edgelist: Graph adjacency list
    :type edgelist: list
    :param dim: Number of nodes in the graph
    :type dim: int
    :param directed: Whether the graph should be treated as directed
    :type directed: bool
    :return: Graph as an adjacency matrix
    :rtype: np.ndarray
    """
    if dim is None:
        dim = get_dim(edgelist)
    A = np.zeros((dim, dim), dtype=np.double)
    for edge in edgelist:
        p, q, wt = edge
        A[p, q] = wt

        if not directed:
            A[q, p] = wt
    return A


def compute_pinverse_diagonal(D):
    D_i = D.copy()
    for i in range(D_i.shape[0]):
        D_i[i, i] = 1 / D[i, i] if D[i, i] != 0 else 0
    return D_i


def compute_X_normalized(A, D, t=-1, lm=1, is_normalized=True):
    D_i = compute_pinverse_diagonal(D)
    P = np.matmul(D_i, A)
    Identity = np.identity(A.shape[0])
    e = np.ones((A.shape[0], 1))

    # Compute W
    scale = np.matmul(e.T, np.matmul(D, e))[0, 0]
    W = np.multiply(1 / scale, np.matmul(e, np.matmul(e.T, D)))

    up_P = np.multiply(lm, P - W)
    X_ = Identity - up_P
    X_i = np.linalg.pinv(X_)

    if t > 0:
        LP_t = Identity - np.linalg.matrix_power(up_P, t)
        X_i = np.matmul(X_i, LP_t)

    if not is_normalized:
        return X_i

    # Normalize with steady state
    SS = np.sqrt(np.matmul(D, e))
    SS = compute_pinverse_diagonal(np.diag(SS.flatten()))

    return np.matmul(X_i, SS)


######################################################


def compute_cw_score(p, q, edgedict, ndict, params=None):
    """
    Computes the common weighted score between p and q.

    :param p: A node of the graph
    :param q: Another node in the graph
    :param edgedict: A dictionary with key `(p, q)` and value `w`.
    :type edgedict: dict
    :param ndict: A dictionary with key `p` and the value a set `{p1, p2, ...}`
    :type ndict: dict
    :param params: Should always be none here
    :type params: None
    :return: A real value representing the score
    :rtype: float
    """
    if len(ndict[p]) > len(ndict[q]):
        temp = p
        p = q
        q = temp
    score = 0
    for elem in ndict[p]:
        if elem in ndict[q]:
            p_elem = (
                edgedict[(p, elem)]
                if (p, elem) in edgedict
                else edgedict[(elem, p)]
            )
            q_elem = (
                edgedict[(q, elem)]
                if (q, elem) in edgedict
                else edgedict[(elem, q)]
            )
            score += p_elem + q_elem
    return score


def compute_cw_score_normalized(p, q, edgedict, ndict, params=None):
    """
    Computes the common weighted normalized score between p and q.

    :param p: A node of the graph
    :param q: Another node in the graph
    :param edgedict: A dictionary with key `(p, q)` and value `w`.
    :type edgedict: dict
    :param ndict: A dictionary with key `p` and the value a set `{p1, p2, ...}`
    :type ndict: dict
    :param params: Should always be none here
    :type params: None
    :return: A real value representing the score
    :rtype: float
    """
    if len(ndict[p]) > len(ndict[q]):
        temp = p
        p = q
        q = temp
    score = 0
    for elem in ndict[p]:
        if elem in ndict[q]:
            p_elem = (
                edgedict[(p, elem)]
                if (p, elem) in edgedict
                else edgedict[(elem, p)]
            )
            q_elem = (
                edgedict[(q, elem)]
                if (q, elem) in edgedict
                else edgedict[(elem, q)]
            )
            score += p_elem + q_elem
    degrees = params["deg"]
    return score / np.sqrt(degrees[p] * degrees[q])


def compute_l3_unweighted_mat(A):
    A_u = np.where(A > 0, 1, 0)
    d, _ = A_u.shape
    e = np.ones((d, 1))
    deg = A_u @ e
    ideg = np.where(deg > 0, 1 / deg, 0)
    sdeg = np.diag(np.sqrt(ideg).flatten())
    A1 = sdeg @ A_u @ sdeg
    return A1


def compute_l3_weighted_mat(A):
    d, _ = A.shape
    e = np.ones((d, 1))
    deg = A @ e
    ideg = np.where(deg > 0, 1 / deg, 0)
    sdeg = np.diag(np.sqrt(ideg).flatten())
    A1 = sdeg @ A @ sdeg
    return A1


def compute_l3_score_mat(p, q, edgedict, ndict, params=None):
    L3 = params["l3"]
    return L3[p, q]


def compute_degree_vec(edgelist):
    A = densify(edgelist)
    e = np.ones((A.shape[0], 1))
    deg = A @ e
    return deg.flatten()


##############################################################


def glide_predict_links(edgelist, X, params={}, thres_p=0.9):
    """Predicts the most likely links in a graph given an embedding X
    of a graph.
    Returns a ranked list of (edges, distances) sorted from closest to
    furthest.

    :param edgelist: A list with elements of type `(p, q, wt)`
    :param X: A nxk embedding matrix
    :param params: A dictionary with entries

    {
        alpha       => real number
        beta        => real number
        delta       => real number
        loc         => String, can be `cw` for common weighted, `l3` for l3 local scoring

        ### To enable ctypes, the following entries should be there ###

        ctypes_on   => True  # This key should only be added if ctypes is on (dont add this
                           # if ctypes is not added)
        so_location => String location of the .so dynamic library

    }
    """
    edgedict = create_edge_dict(edgelist)
    ndict = create_neighborhood_dict(edgelist)
    params_ = {}

    # Embedding
    pairwise_dist = spatial.squareform(spatial.pdist(X))
    N = X.shape[0]
    alpha = params["alpha"]
    local_metric = params["loc"]
    beta = params["beta"]
    delta = params["delta"]
    if local_metric == "l3_u" or local_metric == "l3":
        A = densify(edgelist)
        L3 = compute_l3_unweighted_mat(A)
        params_["l3"] = L3
        local_metric = compute_l3_score_mat
    elif local_metric == "l3_w":
        A = densify(edgelist)
        L3 = compute_l3_weighted_mat(A)
        params_["l3"] = L3
        local_metric = compute_l3_score_mat
    elif local_metric == "cw":
        local_metric = compute_cw_score
    elif local_metric == "cw_normalized":
        params_["deg"] = compute_degree_vec(edgelist)
        local_metric = compute_cw_score_normalized
    else:
        raise Exception("[x] The local scoring metric is not available.")

    glide_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            local_score = local_metric(i, j, edgedict, ndict, params_)
            dsed_dist = pairwise_dist[i, j]
            glide_score = (
                np.exp(alpha / (1 + beta * dsed_dist)) * local_score
                + delta * 1 / dsed_dist
            )
            glide_mat[i, j] = glide_score
            glide_mat[j, i] = glide_score
    if thres_p > 0:
        thres = np.percentile(glide_mat, thres_p)
        for i in range(N):
            for j in range(i):
                glide_mat[i, j] = float(glide_mat[i, j] >= thres)
                glide_mat[j, i] = float(glide_mat[j, i] >= thres)
    return glide_mat


def create_edge_dict(edgelist):
    """
    Creates an edge dictionary with the edge `(p, q)` as the key, and weight `w` as the value.

    :param edgelist: list with elements of form `(p, q, w)`
    :type edgelist: list
    :return: A dictionary with key `(p, q)` and value `w`.
    :rtype: dict
    """
    edgedict = {}
    for (p, q, w) in edgelist:
        edgedict[(p, q)] = w
    return edgedict


def create_neighborhood_dict(edgelist):
    """
    Create a dictionary with nodes as key and a list of neighborhood nodes as the value

    :param edgelist: A list with elements of form `(p, q, w)`
    :type edgelist: list
    :return: neighborhood_dict -> A dictionary with key `p` and value, a set `{p1, p2, p3, ...}`
    :rtype: dict
    """
    ndict = {}
    for ed in edgelist:
        p, q, _ = ed
        if p not in ndict:
            ndict[p] = set()
        if q not in ndict:
            ndict[q] = set()
        ndict[p].add(q)
        ndict[q].add(p)
    return ndict


def glide_compute_map(pos_df, thres_p=0.9, params={}):
    """
    Return glide_mat and glide_map.

    :param pos_df: Dataframe of weighted edges
    :type pos_df: pd.DataFrame
    :param thres_p: Threshold to treat an edge as positive
    :type thres_p: float
    :param params: Parameters for GLIDE
    :type params: dict
    :return: glide_matrix and corresponding glide_map
    :rtype: tuple(np.ndarray, dict)
    """
    params["lam"] = 1 if "lam" not in params else params["lam"]
    params["norm"] = False if "norm" not in params else params["norm"]
    params["glide"] = (
        {"alpha": 1.0, "beta": 1000.0, "loc": "cw_normalized", "delta": 1.0}
        if "glide" not in params
        else params["glide"]
    )

    def a_d(u_edges, n_nodes):
        A = np.zeros((n_nodes, n_nodes))
        for p, q, w in u_edges:
            A[p, q] = w
            A[q, p] = w
        D = np.diag((A @ np.ones((n_nodes, 1))).flatten())
        return A, D

    glide_map = {}
    count = 0
    u_edges = []
    for _, (p, q, w) in pos_df.iterrows():
        for m in [p, q]:
            if m not in glide_map:
                glide_map[m] = count
                count += 1
        u_edges.append((glide_map[p], glide_map[q], w))
    A, D = a_d(u_edges, count)
    X = compute_X_normalized(
        A, D, lm=params["lam"], is_normalized=params["norm"]
    )
    glide_mat = glide_predict_links(
        u_edges, X, params=params["glide"], thres_p=thres_p
    )
    return glide_mat, glide_map


def glider_score(p, q, glider_map, glider_mat):
    for m in [p, q]:
        if m not in glider_map:
            return 0
    p_ = glider_map[p]
    q_ = glider_map[q]
    return glider_mat[p_, q_]
