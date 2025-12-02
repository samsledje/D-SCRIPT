"""
Tests for graph-based link prediction functions in dscript.glider
"""

import numpy as np
import pandas as pd
import pytest

from dscript.glider import (
    compute_X_normalized,
    compute_cw_score,
    compute_cw_score_normalized,
    compute_degree_vec,
    compute_l3_score_mat,
    compute_l3_unweighted_mat,
    compute_l3_weighted_mat,
    compute_pinverse_diagonal,
    create_edge_dict,
    create_neighborhood_dict,
    densify,
    get_dim,
    glide_compute_map,
    glide_predict_links,
    glider_score,
)


class TestGetDim:
    """Tests for get_dim function"""

    def test_get_dim_simple_graph(self):
        """Test getting dimension of a simple graph"""
        edgelist = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        dim = get_dim(edgelist)
        assert dim == 4

    def test_get_dim_disconnected_graph(self):
        """Test getting dimension of a disconnected graph"""
        edgelist = [(0, 1, 1.0), (2, 3, 1.0), (4, 5, 1.0)]
        dim = get_dim(edgelist)
        assert dim == 6

    def test_get_dim_empty_graph(self):
        """Test getting dimension of an empty graph"""
        edgelist = []
        dim = get_dim(edgelist)
        assert dim == 0

    def test_get_dim_single_edge(self):
        """Test getting dimension of a graph with a single edge"""
        edgelist = [(0, 1, 1.0)]
        dim = get_dim(edgelist)
        assert dim == 2

    def test_get_dim_self_loop(self):
        """Test getting dimension with self loops"""
        edgelist = [(0, 0, 1.0), (0, 1, 1.0)]
        dim = get_dim(edgelist)
        assert dim == 2


class TestDensify:
    """Tests for densify function"""

    def test_densify_simple_graph(self):
        """Test densifying a simple graph"""
        edgelist = [(0, 1, 1.0), (1, 2, 1.5)]
        A = densify(edgelist, dim=3, directed=False)

        assert A.shape == (3, 3)
        assert A[0, 1] == 1.0
        assert A[1, 0] == 1.0  # Undirected
        assert A[1, 2] == 1.5
        assert A[2, 1] == 1.5  # Undirected
        assert A[0, 2] == 0.0  # No edge

    def test_densify_directed_graph(self):
        """Test densifying a directed graph"""
        edgelist = [(0, 1, 1.0), (1, 0, 2.0)]
        A = densify(edgelist, dim=2, directed=True)

        assert A[0, 1] == 1.0
        assert A[1, 0] == 2.0

    def test_densify_auto_dim(self):
        """Test densify with automatic dimension detection"""
        edgelist = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        A = densify(edgelist)

        assert A.shape == (4, 4)

    def test_densify_empty_graph(self):
        """Test densifying an empty graph"""
        edgelist = []
        A = densify(edgelist, dim=3)

        assert A.shape == (3, 3)
        assert np.allclose(A, 0.0)

    def test_densify_weighted_graph(self):
        """Test densifying a weighted graph"""
        edgelist = [(0, 1, 0.5), (1, 2, 0.3), (0, 2, 0.8)]
        A = densify(edgelist, dim=3)

        assert A[0, 1] == 0.5
        assert A[1, 2] == 0.3
        assert A[0, 2] == 0.8


class TestComputePinverseDiagonal:
    """Tests for compute_pinverse_diagonal function"""

    def test_compute_pinverse_diagonal_basic(self):
        """Test computing pseudo-inverse of diagonal matrix"""
        D = np.diag([1.0, 2.0, 4.0])
        D_i = compute_pinverse_diagonal(D)

        expected = np.array([[1.0, 0, 0], [0, 0.5, 0], [0, 0, 0.25]])
        assert np.allclose(D_i, expected)

    def test_compute_pinverse_diagonal_with_zero(self):
        """Test computing pseudo-inverse with zero diagonal element"""
        D = np.diag([1.0, 0.0, 4.0])
        D_i = compute_pinverse_diagonal(D)

        expected = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0.25]])
        assert np.allclose(D_i, expected)

    def test_compute_pinverse_diagonal_identity(self):
        """Test computing pseudo-inverse of identity matrix"""
        D = np.eye(3)
        D_i = compute_pinverse_diagonal(D)

        assert np.allclose(D_i, np.eye(3))

    def test_compute_pinverse_diagonal_off_diagonal(self):
        """Test that off-diagonal elements are preserved"""
        D = np.array([[2.0, 1.0, 0.5], [1.0, 3.0, 0.8], [0.5, 0.8, 4.0]])
        D_i = compute_pinverse_diagonal(D)

        # Only diagonal should be inverted
        assert D_i[0, 0] == 0.5
        assert D_i[1, 1] == 1.0 / 3.0
        assert D_i[2, 2] == 0.25
        # Off-diagonal should remain the same
        assert D_i[0, 1] == 1.0
        assert D_i[1, 0] == 1.0


class TestComputeXNormalized:
    """Tests for compute_X_normalized function"""

    def test_compute_x_normalized_basic(self):
        """Test computing normalized X matrix"""
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        D = np.diag([1, 2, 1])

        X = compute_X_normalized(A, D)

        # Basic shape check
        assert X.shape == A.shape

    def test_compute_x_normalized_identity(self):
        """Test with identity matrix"""
        A = np.eye(3)
        D = np.eye(3)

        X = compute_X_normalized(A, D)
        assert X.shape == (3, 3)

    def test_compute_x_normalized_with_t(self):
        """Test compute_X_normalized with t parameter"""
        A = np.array([[0, 1], [1, 0]], dtype=float)
        D = np.diag([1, 1])

        X = compute_X_normalized(A, D, t=2)
        assert X.shape == (2, 2)

    def test_compute_x_normalized_not_normalized(self):
        """Test compute_X_normalized without normalization"""
        A = np.array([[0, 1], [1, 0]], dtype=float)
        D = np.diag([1, 1])

        X = compute_X_normalized(A, D, is_normalized=False)
        assert X.shape == (2, 2)


class TestCreateEdgeDict:
    """Tests for create_edge_dict function"""

    def test_create_edge_dict_basic(self):
        """Test creating edge dictionary"""
        edgelist = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]
        edgedict = create_edge_dict(edgelist)

        assert edgedict[(0, 1)] == 1.0
        assert edgedict[(1, 2)] == 2.0
        assert edgedict[(0, 2)] == 3.0
        assert len(edgedict) == 3

    def test_create_edge_dict_empty(self):
        """Test creating edge dictionary from empty list"""
        edgelist = []
        edgedict = create_edge_dict(edgelist)

        assert len(edgedict) == 0

    def test_create_edge_dict_duplicate_edges(self):
        """Test creating edge dictionary with duplicate edges (last one wins)"""
        edgelist = [(0, 1, 1.0), (0, 1, 2.0)]
        edgedict = create_edge_dict(edgelist)

        # Last edge should win
        assert edgedict[(0, 1)] == 2.0


class TestCreateNeighborhoodDict:
    """Tests for create_neighborhood_dict function"""

    def test_create_neighborhood_dict_basic(self):
        """Test creating neighborhood dictionary"""
        edgelist = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        ndict = create_neighborhood_dict(edgelist)

        assert 1 in ndict[0]
        assert 2 in ndict[0]
        assert 0 in ndict[1]
        assert 2 in ndict[1]
        assert 0 in ndict[2]
        assert 1 in ndict[2]

    def test_create_neighborhood_dict_empty(self):
        """Test creating neighborhood dictionary from empty list"""
        edgelist = []
        ndict = create_neighborhood_dict(edgelist)

        assert len(ndict) == 0

    def test_create_neighborhood_dict_isolated_nodes(self):
        """Test creating neighborhood dictionary with multiple components"""
        edgelist = [(0, 1, 1.0), (2, 3, 1.0)]
        ndict = create_neighborhood_dict(edgelist)

        assert 1 in ndict[0]
        assert 0 in ndict[1]
        assert 3 in ndict[2]
        assert 2 in ndict[3]
        assert len(ndict) == 4

    def test_create_neighborhood_dict_self_loop(self):
        """Test neighborhood dict with self loops"""
        edgelist = [(0, 0, 1.0), (0, 1, 1.0)]
        ndict = create_neighborhood_dict(edgelist)

        assert 0 in ndict[0]
        assert 1 in ndict[0]


class TestComputeCWScore:
    """Tests for compute_cw_score function"""

    def test_compute_cw_score_basic(self):
        """Test computing common weighted score"""
        edgelist = [(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)]
        edgedict = create_edge_dict(edgelist)
        ndict = create_neighborhood_dict(edgelist)

        # Nodes 0 and 1 have common neighbor 2
        score = compute_cw_score(0, 1, edgedict, ndict)

        # Should sum weights: 2.0 (0->2) + 3.0 (1->2) = 5.0
        assert score == 5.0

    def test_compute_cw_score_no_common_neighbors(self):
        """Test CW score when nodes have no common neighbors"""
        edgelist = [(0, 1, 1.0), (2, 3, 1.0)]
        edgedict = create_edge_dict(edgelist)
        ndict = create_neighborhood_dict(edgelist)

        score = compute_cw_score(0, 2, edgedict, ndict)
        assert score == 0.0

    def test_compute_cw_score_multiple_common(self):
        """Test CW score with multiple common neighbors"""
        edgelist = [(0, 2, 1.0), (0, 3, 2.0), (1, 2, 3.0), (1, 3, 4.0)]
        edgedict = create_edge_dict(edgelist)
        ndict = create_neighborhood_dict(edgelist)

        # Nodes 0 and 1 have common neighbors 2 and 3
        score = compute_cw_score(0, 1, edgedict, ndict)

        # Should sum: (1.0 + 3.0) + (2.0 + 4.0) = 10.0
        assert score == 10.0


class TestComputeCWScoreNormalized:
    """Tests for compute_cw_score_normalized function"""

    def test_compute_cw_score_normalized_basic(self):
        """Test computing normalized common weighted score"""
        edgelist = [(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)]
        edgedict = create_edge_dict(edgelist)
        ndict = create_neighborhood_dict(edgelist)
        degrees = compute_degree_vec(edgelist)

        params = {"deg": degrees}
        score = compute_cw_score_normalized(0, 1, edgedict, ndict, params)

        # Should normalize by sqrt(deg[0] * deg[1])
        unnormalized = 5.0  # From previous test
        expected = unnormalized / np.sqrt(degrees[0] * degrees[1])
        assert np.isclose(score, expected)

    def test_compute_cw_score_normalized_no_common(self):
        """Test normalized CW score with no common neighbors"""
        edgelist = [(0, 1, 1.0), (2, 3, 1.0)]
        edgedict = create_edge_dict(edgelist)
        ndict = create_neighborhood_dict(edgelist)
        degrees = compute_degree_vec(edgelist)

        params = {"deg": degrees}
        score = compute_cw_score_normalized(0, 2, edgedict, ndict, params)

        assert score == 0.0


class TestComputeL3Functions:
    """Tests for L3 matrix computation functions"""

    def test_compute_l3_unweighted_mat_basic(self):
        """Test computing L3 unweighted matrix"""
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        L3 = compute_l3_unweighted_mat(A)

        assert L3.shape == A.shape
        # L3 should be normalized
        assert np.all(L3 >= 0)

    def test_compute_l3_weighted_mat_basic(self):
        """Test computing L3 weighted matrix"""
        A = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)
        L3 = compute_l3_weighted_mat(A)

        assert L3.shape == A.shape
        assert np.all(L3 >= 0)

    def test_compute_l3_unweighted_vs_weighted(self):
        """Test that unweighted and weighted L3 differ for weighted graphs"""
        A = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)

        L3_u = compute_l3_unweighted_mat(A)
        L3_w = compute_l3_weighted_mat(A)

        # They should be different
        assert not np.allclose(L3_u, L3_w)

    def test_compute_l3_score_mat(self):
        """Test computing L3 score from matrix"""
        L3 = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.8], [0.2, 0.8, 1.0]])
        params = {"l3": L3}

        score = compute_l3_score_mat(0, 1, None, None, params)
        assert score == 0.5

        score = compute_l3_score_mat(1, 2, None, None, params)
        assert score == 0.8


class TestComputeDegreeVec:
    """Tests for compute_degree_vec function"""

    def test_compute_degree_vec_basic(self):
        """Test computing degree vector"""
        edgelist = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        deg = compute_degree_vec(edgelist)

        # Each node has degree 2 (undirected)
        assert len(deg) == 3
        assert np.allclose(deg, [2.0, 2.0, 2.0])

    def test_compute_degree_vec_weighted(self):
        """Test computing degree vector with weights"""
        edgelist = [(0, 1, 2.0), (1, 2, 3.0)]
        deg = compute_degree_vec(edgelist)

        assert len(deg) == 3
        assert deg[0] == 2.0
        assert deg[1] == 5.0  # 2.0 + 3.0
        assert deg[2] == 3.0

    def test_compute_degree_vec_isolated_node(self):
        """Test degree vector with various degrees"""
        edgelist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)]
        deg = compute_degree_vec(edgelist)

        assert deg[0] == 3.0  # Hub node
        assert deg[1] == 1.0
        assert deg[2] == 1.0
        assert deg[3] == 1.0


class TestGlidePredictLinks:
    """Tests for glide_predict_links function"""

    def test_glide_predict_links_cw(self):
        """Test GLIDE link prediction with common weighted scoring"""
        edgelist = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        X = np.array([[0, 0], [1, 0], [0.5, 0.5]])
        params = {"alpha": 1.0, "beta": 1.0, "loc": "cw", "delta": 0.1}

        glide_mat = glide_predict_links(edgelist, X, params, thres_p=0)

        assert glide_mat.shape == (3, 3)
        # Matrix should be symmetric
        assert np.allclose(glide_mat, glide_mat.T)
        # Diagonal should be zero
        assert np.allclose(np.diag(glide_mat), 0)

    def test_glide_predict_links_l3(self):
        """Test GLIDE with L3 scoring"""
        edgelist = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        X = np.array([[0, 0], [1, 0], [0.5, 0.5]])
        params = {"alpha": 1.0, "beta": 1.0, "loc": "l3", "delta": 0.1}

        glide_mat = glide_predict_links(edgelist, X, params, thres_p=0)

        assert glide_mat.shape == (3, 3)
        assert np.allclose(glide_mat, glide_mat.T)

    def test_glide_predict_links_l3_weighted(self):
        """Test GLIDE with L3 weighted scoring"""
        edgelist = [(0, 1, 2.0), (1, 2, 3.0), (2, 0, 1.5)]
        X = np.array([[0, 0], [1, 0], [0.5, 0.5]])
        params = {"alpha": 1.0, "beta": 1.0, "loc": "l3_w", "delta": 0.1}

        glide_mat = glide_predict_links(edgelist, X, params, thres_p=0)

        assert glide_mat.shape == (3, 3)

    def test_glide_predict_links_cw_normalized(self):
        """Test GLIDE with normalized CW scoring"""
        edgelist = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        X = np.array([[0, 0], [1, 0], [0.5, 0.5]])
        params = {"alpha": 1.0, "beta": 1.0, "loc": "cw_normalized", "delta": 0.1}

        glide_mat = glide_predict_links(edgelist, X, params, thres_p=0)

        assert glide_mat.shape == (3, 3)

    def test_glide_predict_links_with_threshold(self):
        """Test GLIDE with thresholding"""
        edgelist = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        X = np.array([[0, 0], [1, 0], [0.5, 0.5]])
        params = {"alpha": 1.0, "beta": 1.0, "loc": "cw", "delta": 0.1}

        glide_mat = glide_predict_links(edgelist, X, params, thres_p=0.5)

        # With threshold, values should be 0 or 1
        unique_vals = np.unique(glide_mat)
        assert len(unique_vals) <= 2
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_glide_predict_links_invalid_metric(self):
        """Test GLIDE with invalid metric raises exception"""
        edgelist = [(0, 1, 1.0), (1, 2, 1.0)]
        X = np.array([[0, 0], [1, 0], [0.5, 0.5]])
        params = {"alpha": 1.0, "beta": 1.0, "loc": "invalid", "delta": 0.1}

        with pytest.raises(Exception) as exc_info:
            glide_predict_links(edgelist, X, params)
        assert "not available" in str(exc_info.value)


class TestGlideComputeMap:
    """Tests for glide_compute_map function"""

    def test_glide_compute_map_basic(self):
        """Test computing glide map from dataframe"""
        df = pd.DataFrame(
            {"protein1": ["A", "B", "C"], "protein2": ["B", "C", "A"], "weight": [1.0, 1.0, 1.0]}
        )

        params = {"glide": {"alpha": 1.0, "beta": 1.0, "loc": "cw", "delta": 0.1}}
        glide_mat, glide_map = glide_compute_map(df, thres_p=0, params=params)

        # Should have 3 nodes
        assert len(glide_map) == 3
        assert glide_mat.shape == (3, 3)

    def test_glide_compute_map_default_params(self):
        """Test glide_compute_map with default parameters"""
        df = pd.DataFrame(
            {"protein1": ["A", "B"], "protein2": ["B", "C"], "weight": [1.0, 1.0]}
        )

        glide_mat, glide_map = glide_compute_map(df)

        assert len(glide_map) == 3
        assert glide_mat.shape == (3, 3)

    def test_glide_compute_map_creates_mapping(self):
        """Test that glide_compute_map creates correct protein mapping"""
        df = pd.DataFrame(
            {
                "protein1": ["P1", "P2", "P1"],
                "protein2": ["P2", "P3", "P3"],
                "weight": [1.0, 1.0, 1.0],
            }
        )

        _, glide_map = glide_compute_map(df, thres_p=0)

        # All proteins should be in the map
        assert "P1" in glide_map
        assert "P2" in glide_map
        assert "P3" in glide_map
        # Mapping should be to sequential integers
        assert set(glide_map.values()) == {0, 1, 2}


class TestGliderScore:
    """Tests for glider_score function"""

    def test_glider_score_basic(self):
        """Test getting glider score for a pair"""
        glider_map = {"A": 0, "B": 1, "C": 2}
        glider_mat = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.8], [0.2, 0.8, 1.0]])

        score = glider_score("A", "B", glider_map, glider_mat)
        assert score == 0.5

        score = glider_score("B", "C", glider_map, glider_mat)
        assert score == 0.8

    def test_glider_score_missing_node(self):
        """Test glider score when node not in map"""
        glider_map = {"A": 0, "B": 1}
        glider_mat = np.array([[1.0, 0.5], [0.5, 1.0]])

        score = glider_score("A", "C", glider_map, glider_mat)
        assert score == 0

        score = glider_score("X", "Y", glider_map, glider_mat)
        assert score == 0

    def test_glider_score_symmetric(self):
        """Test that glider score is symmetric"""
        glider_map = {"A": 0, "B": 1, "C": 2}
        glider_mat = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.8], [0.2, 0.8, 1.0]])

        score_ab = glider_score("A", "B", glider_map, glider_mat)
        score_ba = glider_score("B", "A", glider_map, glider_mat)

        assert score_ab == score_ba

    def test_glider_score_self(self):
        """Test glider score for node with itself"""
        glider_map = {"A": 0, "B": 1}
        glider_mat = np.array([[1.0, 0.5], [0.5, 1.0]])

        score = glider_score("A", "A", glider_map, glider_mat)
        assert score == 1.0
