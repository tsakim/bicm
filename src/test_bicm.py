# -*- coding: utf-8 -*-
"""
Test file for the module bicm

Created on Tue Aug 21, 2017

Author:
    Mika Straka

Description:
    This file contains the test cases for the functions and methods
    defined in bicm.py. The tests can be run with ``pytest``.

Usage:
    To run the tests, execute
        $ pytest test_bicm.py
    in the command line. If you want to run the tests in verbose mode, use
        $ pytest test_bicm.py -v
    or
        $ pytest test_bicm.py -v  -r P
    to capture the output of the SciPy solver.

Note that bicm.py and test_bicm.py should be in the same directory.
"""

################################################################################
# Tests
################################################################################

import numpy as np
from bicm import BiCM


# BiCM.make_bicm() -------------------------------------------------------------

def test_tutorial_matrix():
    """Test the matrix for the tutorial."""
    mat = np.array([[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1]])
    cm = BiCM(mat)
    cm.make_bicm(method='hybr')
    assert cm.test_average_degrees(eps=1e-12)
    cm.make_bicm(method='lm', options={'maxiter': 2000})
    assert cm.test_average_degrees(eps=1e-12)


def test_make_bicm():
    """Test the correct BiCM biadjacency matrices are created."""
    td = np.array([[1, 0], [0, 1]])
    exp_adjmat = np.array([[0.5, 0.5], [0.5, 0.5]])
    cm = BiCM(td)
    cm.make_bicm()
    assert np.all(np.abs(cm.adj_matrix - exp_adjmat) < 1e-12)

    td = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    exp_adjmat = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
    cm = BiCM(td)
    cm.make_bicm()
    assert np.all(np.abs(cm.adj_matrix - exp_adjmat) < 1e-12)


def test_solver_convergence():
    """Test different convergence of solvers.

    In this example, using the standard solver options the algorithm converges
    when least-squares 'lm' is used, whereas it does not when 'hybr' is applied.

    Run

    .. code::

        $ pytest test_bicm.py -v  -r P

    to see and compare the solver output.
    """
    td = np.array([[1, 0], [1, 1], [1, 1]])
    cm = BiCM(td)

    # 'hybr' does not converge:
    cm.make_bicm(method='hybr')
    assert not cm.sol.success

    # 'lm' does converge:
    cm.make_bicm(method='lm')
    assert cm.sol.success

    # degree differences are between 1e-7 and 1-7
    assert np.all(np.abs(cm.adj_matrix - td) < 1e-7)
    assert np.any(np.abs(cm.adj_matrix - td) > 1e-8)

# BiCM.solve_equations ---------------------------------------------------------

def test_solve_equations():
    """Test the solving function."""
    cm = BiCM(np.array([[1, 0], [0, 1]]))
    exp_adjmat = np.array([[0.5, 0.5], [0.5, 0.5]])
    sol = cm.solve_equations()
    assert np.all(np.abs(cm.get_biadjacency_matrix(sol.x) - exp_adjmat) < 1e-12)


# BiCM.equation ----------------------------------------------------------------

def test_equations():
    """Check that the correct values are returned from the equation system. """
    cm = BiCM(np.array([[1, 0, 1], [1, 1, 1]]))
    exp_eq = np.array([-1.83984534, -2.6970858, -1.88746439,
                       -0.86147186, -1.78799489])
    eq = cm.equations((np.array([0.1, 0.2, 0.4, 0.5, 0.8])))
    assert np.all(np.abs(exp_eq - eq) < 1e-8)


# BiCM.jacobian ----------------------------------------------------------------

def test_jacobian():
    """Test that the correct Jacobian expressions are returned."""
    # TODO check expression of Jacobian and precision - seems to be a bit small
    cm = BiCM(np.array([[1, 0, 1], [1, 1, 1]]))
    exp_jac = np.array([[1.51, 0., 0.09, 0.09, 0.09],
                        [0., 1.35, 0.17, 0.17, 0.15],
                        [0.37, 0.34, 0.26, 0., 0.],
                        [0.45, 0.41, 0., 0.26, 0.],
                        [0.69, 0.59, 0., 0., 0.23]])
    jac = cm.jacobian((np.array([0.1, 0.2, 0.4, 0.5, 0.8]))),
    assert np.all(np.abs(exp_jac - jac) < 1e-2)


# BiCM.get_biadjacency_matrix --------------------------------------------------

def test_get_biadjacency_matrix():
    """Test that the correct biadjacency matrix is returned."""
    cm = BiCM(np.array([[1, 0, 1], [1, 1, 1]]))
    exp_adj = np.array([[0.03846154, 0.04761905, 0.07407407],
                        [0.07407407, 0.09090909, 0.13793103]])
    adj = cm.get_biadjacency_matrix((np.array([0.1, 0.2, 0.4, 0.5, 0.8])))
    assert np.all(np.abs(exp_adj - adj) < 1e-8)


# BiCM.get_triup_dim -----------------------------------------------------------

def test_get_triup_dim():
    """Check that the function returns the correct number of nodes couples."""
    cm = BiCM(np.array([[1, 0, 1], [1, 1, 1]]))
    assert cm.get_triup_dim(False) == 3
    assert cm.get_triup_dim(True) == 1

    td = np.random.randint(low=0, high=2, size=50).reshape(5, 10)
    cm = BiCM(td)
    n = cm.get_triup_dim(True)
    assert n == td.shape[0] * (td.shape[0] - 1) / 2
    n = cm.get_triup_dim(False)
    assert n == td.shape[1] * (td.shape[1] - 1) / 2


# BiCM.get_lambda_motif_block --------------------------------------------------

def test_get_lambda_motif_block():
    """Test that the correct subblock of the matrix is returned."""
    binmat = np.array([[1, 1, 0], [1, 1, 1]]).T
    cm = BiCM(binmat)

    # Lambda motifs of countries:
    assert np.all(cm.get_lambda_motif_block(binmat, 0, 2) ==
                  np.array([2., 1., 1.]))


# BiCM.get_pvalues -------------------------------------------------------------

def test_get_pvalues():
    """Check that the correct p-values are returned."""
    binmat = np.array([[1, 1, 0], [1, 1, 1]]).T
    cm = BiCM(binmat)

    p = [0.1, 0.5, 0.7]
    plambm = np.array([[p, p, p], [p, p, p], [p, p, p]])
    nlambm = np.array([2, 1, 1])
    assert np.all(np.abs(cm.get_pvalues_q(plambm[0], nlambm, parallel=False,
                                          k1=0, k2=3)
                  - np.array([0.4, 0.865, 0.865])) < 1e-12)

    # Check that sequential and parallel processing obtain the same results:
    pv_seq = cm.get_pvalues_q(plambm[0], nlambm, parallel=False, k1=0, k2=3)
    pv_par = cm.get_pvalues_q(plambm[0], nlambm, parallel=True, k1=0, k2=3)
    assert np.all(pv_seq == pv_par)


# BiCM.trium2flat_idx ----------------------------------------------------------

def test_trium2flat_idx():
    """Test the conversion from triangular upper index to flat array index."""
    td = np.array([[1, 0], [0, 1], [0, 1]])
    cm = BiCM(td)
    assert cm.triumat2flat_idx(3, 5, 8) == 19
    for k in range(45):
        ij = cm.flat2triumat_idx(k, 10)
        assert cm.triumat2flat_idx(ij[0], ij[1], 10) == k
    for i in range(10):
        for j in range(i + 1, 10):
            k = cm.triumat2flat_idx(i, j, 10)
            assert (i, j) == cm.flat2triumat_idx(k, 10)


# BiCM.split_range -------------------------------------------------------------

def test_split_range():
    """Test that an array is split into the correct pieces."""
    td = np.random.randint(low=0, high=2, size=50).reshape(5, 10)
    cm = BiCM(td)
    n = cm.get_triup_dim(True)
    assert n == td.shape[0] * (td.shape[0] - 1) / 2
    kk = cm.split_range(n, m=5)
    assert kk == [i * n / 5 for i in range(5)]


# BiCM.get_lambda_motif_block --------------------------------------------------

def test_get_lambda_motif_block():
    """Test that the correct numbers of lambda motifs are obtained."""
    td = np.random.randint(low=0, high=2, size=123000).reshape(123, 1000)
    cm= BiCM(td)
    n = cm.get_triup_dim(True)
    nl = np.dot(td, td.T)[np.triu_indices(n=td.shape[0], k=1)]
    k1 = np.random.randint(low=0, high=n/2)
    k2 = np.random.randint(low=n/2, high=n)
    nl2 = cm.get_lambda_motif_block(cm.bin_mat, k1, k2)
    assert np.all((nl[k1:k2] == nl2))
    k1 = 0
    k2 = n - 1
    nl3 = cm.get_lambda_motif_block(cm.bin_mat, k1, k2)
    if not len(nl) ==  len(nl3):
        print n, len(nl),  len(nl3)
    assert np.all(nl[k1:k2] == nl3[k1:k2])

    k1 = np.random.randint(low=0, high=n - 1)
    k2 = n - 1
    nl3 = cm.get_lambda_motif_block(cm.bin_mat, k1, k2)
    if not np.all(nl[k1:] == nl3):
        print k1, k2, len(nl[k1:]), len(nl3)

    r = np.random.randint(low=10, high=500)
    c = np.random.randint(low=100, high=5000)
    td = np.random.randint(low=0, high=2, size=r * c).reshape(r, c)

    cm = BiCM(td)
    n = cm.get_triup_dim(True)
    nl = np.dot(td, td.T)[np.triu_indices(n=td.shape[0], k=1)]
    k1 = np.random.randint(low=0, high=n/2)
    k2 = np.random.randint(low=n/2, high=n)
    nl2 = cm.get_lambda_motif_block(cm.bin_mat, k1, k2)
    assert np.all((nl[k1:k2] == nl2))


# BiCM.get_plambda_block -------------------------------------------------------

def test_get_plambda_block():
    """Test that the correct lambda-motif probabilities are obtained."""
    td = np.random.randint(low=0, high=2, size=50).reshape(5, 10)
    adj = np.random.random(size=td.size).reshape(td.shape)
    cm = BiCM(td)
    n = cm.get_triup_dim(True)
    plam = np.ones(shape=(n, cm.bin_mat.shape[1])) * (99)
    m = 0;
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            plam[m, :] = adj[i, :] * adj[j, :]
            m += 1
    k1 = 0  # np.random.randint(low=0, high=n/2)
    k2 = 4  # np.random.randint(low=n/2, high=n)
    pl = cm.get_plambda_block(adj, k1, k2)
    assert np.all(plam[k1:k2] == pl)

    k1 = np.random.randint(low=0, high=n/2)
    k2 = np.random.randint(low=n/2, high=n - 1)
    pl = cm.get_plambda_block(adj, k1, k2)
    assert np.all(plam[k1:k2] == pl)

    k1 = 0
    k2 = n - 1
    pl = cm.get_plambda_block(adj, k1, k2)
    assert np.all(plam[k1:] == pl)

    k1 = np.random.randint(low=0, high=n/2)
    k2 = n - 1
    pl = cm.get_plambda_block(adj, k1, k2)
    assert np.all(plam[k1:] == pl)

    # random shape of the input matrix
    r = np.random.randint(low=5, high=15)
    c = np.random.randint(low=10, high=100)
    td = np.random.randint(low=0, high=2, size=r * c).reshape(r, c)
    adj = np.random.random(size=td.size).reshape(td.shape)
    cm = BiCM(td)
    n = cm.get_triup_dim(True)
    plam = np.ones(shape=(n, cm.bin_mat.shape[1])) * 99
    m = 0;
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            plam[m, :] = adj[i, :] * adj[j, :]
            m += 1
    k1 = np.random.randint(low=0, high=n/2)
    k2 = np.random.randint(low=n/2, high=n - 1)
    pl = cm.get_plambda_block(adj, k1, k2)
    assert np.all(plam[k1:k2] == pl)
