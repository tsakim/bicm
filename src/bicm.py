# -*- coding: utf-8 -*-
"""
Created on Tue Dec 1 08:04:28 2015

Module:
    bicm - Bipartite Configuration Model

Author:
    Mika Straka

Description:
    Implementation of the Bipartite Configuration Model (BiCM) for binary
    undirected bipartite networks [Saracco2015]_.

    Given the biadjacency matrix of a bipartite graph in the form of a binary
    array as input, the module allows the user to calculate the biadjacency
    matrix of the ensemble average graph :math:`<G>^*` of the BiCM null model.
    The matrix entries correspond to the link probabilities :math:`<G>^*_{rc} =
    p_{rc}` between nodes of the two distinct bipartite node sets.
    Subsequently, one can calculate the p-values of the node similarities for
    nodes in the same bipartite layer [Saracco2016]_.

Usage:
    Be ``mat`` a two-dimensional binary NumPy array. The nodes of the two
    bipartite layers are ordered along the rows and columns, respectively. In
    the algorithm, the two layers are identified by the boolean values ``True``
    for the **row-nodes** and ``False`` for the **column-nodes**.

    Import the module and initialize the Bipartite Configuration Model::

        >>> from src.bicm import BiCM
        >>> cm = BiCM(bin_mat=mat)

    To create the biadjacency matrix of the BiCM, use::

        >>> cm.make_bicm()

    The biadjacency matrix of the BiCM null model can be saved in *<filename>*::

        >>> cm.save_biadjacency(filename=<filename>, delim='\t')

    By default, the file is saved in a human-readable ``.csv`` format with tab
    delimiters, which can be changed using the keyword ``delim``. The
    information can also be saved as a binary NumPy file ``.npy`` by using::

        >>> cm.save_biadjacency(filename=<filename>, binary=True)

    If the file is not binary, it should end with ``.csv``. If it is binary
    instead, NumPy automatically attaches the ending ``.npy``.


    In order to analyze the similarity of the **row-nodes** and to save
    the p-values of the corresponding :math:`\\Lambda`-motifs (i.e. of the
    number of shared neighbors [Saracco2016]_), use::

        >>> cm.lambda_motifs(True, filename=<filename>)

    For the **column-nodes**, use::

        >>> cm.lambda_motifs(False, filename=<filename>)

    By default, the resulting p-values are saved as binary NumPy file to reduce
    the required disk space, and the format suffix ``.npy`` is appended. If the
    file should be saved in a human-readable ``.csv`` format, use::

        >>> cm.lambda_motifs(True, filename=<filename>, delim='\\t', \
                binary=False)

    or analogously::

        >>> cm.lambda_motifs(False, filename=<filename>, delim='\\t', \
                binary=False)

    .. note::

         The p-values are saved as a one-dimensional array with index
         :math:`k \\in \\left[0, \\ldots, \\binom{N}{2} - 1\\right]` for a
         bipartite layer of :math:`N` nodes. The indices ``(i, j)`` of the
         nodes corresponding to entry ``k`` in the array can be reconstructed
         using the method :func:`BiCM.flat2_triumat_idx`. The number of nodes
         ``N`` can be recovered from the length of the array with
         :func:`BiCM.flat2_triumat_dim`.

    Subsequently, the p-values can be used to perform a multiple hypotheses
    testing of the node similarities and to obtain statistically validated
    monopartite projections [Saracco2016]_. The p-values are calculated in
    parallel by default, see :ref:`parallel` for details.

    .. note::

        Since the calculation of the p-values is computationally demanding, the
        ``bicm`` module uses the Python `multiprocessing
        <https://docs.python.org/2/library/multiprocessing.html>`_ package by
        default for this purpose. The number of parallel processes depends on
        the number of CPUs of the work station (see variable ``num_procs`` in
        the method :func:`BiCM.get_pvalues_q`).

        If the calculation should **not** be performed in parallel, use::

            >>> cm.lambda_motifs(<bool>, parallel=False)

        instead of::

            >>> cm.lambda_motifs(<bool>)

References:
    [Saracco2015] F. Saracco, R. Di Clemente, A. Gabrielli, T. Squartini,
    Randomizing bipartite networks: the case of the World Trade Web, Scientific
    Reports 5, 10595 (2015)

    [Saracco2016] F. Saracco, M. J. Straka, R. Di Clemente, A. Gabrielli, G.
    Caldarelli, T. Squartini, Inferring monopartite projections of bipartite
    networks: an entropy-based approach, arXiv preprint arXiv:1607.02481
"""

import ctypes
import multiprocessing
import scipy.optimize as opt
import numpy as np
from poibin.poibin import PoiBin


class BiCM:
    """Bipartite Configuration Model for undirected binary bipartite networks.

    This class implements the Bipartite Configuration Model (BiCM), which can
    be used as a null model for the analysis of undirected and binary bipartite
    networks. The class provides methods for calculating the biadjacency matrix
    of the null model and for quantifying node similarities in terms of
    p-values.
    """

    def __init__(self, bin_mat):
        """Initialize the parameters of the BiCM.

        :param bin_mat: binary input matrix describing the biadjacency matrix
                of a bipartite graph with the nodes of one layer along the rows
                and the nodes of the other layer along the columns.
        :type bin_mat: numpy.array
        """
        self.bin_mat = np.array(bin_mat, dtype=np.int64)
        self.check_input_matrix_is_binary()
        [self.num_rows, self.num_columns] = self.bin_mat.shape
        self.dseq = self.set_degree_seq()
        self.dim = self.dseq.size
        self.sol = None             # solution of the equation system
        self.adj_matrix = None      # biadjacency matrix of the null model
        self.input_queue = None     # queue for parallel processing
        self.output_queue = None    # queue for parallel processing

    def check_input_matrix_is_binary(self):
        """Check that the input matrix is binary, i.e. entries are 0 or 1.

        :raise AssertionError: raise an error if the input matrix is not
            binary
        """
        assert np.all(np.logical_or(self.bin_mat == 0, self.bin_mat == 1)), \
            "Input matrix is not binary."

    def set_degree_seq(self):
        """Return the node degree sequence of the input matrix.

        :returns: node degree sequence [degrees row-nodes, degrees column-nodes]
        :rtype: numpy.array

        :raise AssertionError: raise an error if the length of the returned
            degree sequence does not correspond to the total number of nodes
        """
        dseq = np.empty(self.num_rows + self.num_columns)
        dseq[self.num_rows:] = np.squeeze(np.sum(self.bin_mat, axis=0))
        dseq[:self.num_rows] = np.squeeze(np.sum(self.bin_mat, axis=1))
        assert dseq.size == (self.num_rows + self.num_columns)
        return dseq

    def make_bicm(self, x0=None, method='hybr', jac=None, tol=None, **kwargs):
        """Create the biadjacency matrix of the BiCM null model.

        Solve the log-likelihood maximization problem to obtain the BiCM
        null model which respects constraints on the degree sequence of the
        input matrix.

        The problem is solved using ``scipy``'s root function with the solver
        defined by ``method``. The status of the solver after running
        ``scipy.root``and the difference between the network and BiCM degrees
        are printed in the console.

        The default solver is the modified Powell method ``hybr``. Least-squares
        can be chosen with ``method='lm'`` for the Levenberg-Marquardt approach.

        Depending on the solver, keyword arguments ``kwargs`` can be passed to
        the solver. Please refer to the `scipy.optimize.root documentation
        <https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/
        scipy.optimize.root.html>`_ for detailed descriptions.

        .. note::

            It can happen that the solver ``method`` used by ``scipy.root``
            does not converge to a solution.
            In this case, please try another ``method`` or different initial
            conditions and refer to the `scipy.optimize.root documentation
            <https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/
            scipy.optimize.root.html>`_.

        :param x0: initial guesses for the solutions.
        :type x0: numpy.array, optional
        :param method: type of solver, default is ‘hybr’. For other
            solvers, see the `scipy.optimize.root documentation
            <https://docs.scipy.org/doc/
            scipy-0.19.0/reference/generated/scipy.optimize.root.html>`_.
        :type method: str, optional
        :param jac: Jacobian of the system
        :type jac: bool or callable, optional
        :param tol: tolerance for termination. For detailed control, use
            solver-specific options.
        :type tol: float, optional
        :param kwargs: solver-specific options, please refer to the SciPy
            documentation

        :raise ValueError: raise an error if not enough initial conditions
            are provided
        """
        self.sol = self.solve_equations(x0=x0, method=method, jac=jac, tol=tol,
                                        **kwargs)
        # create BiCM biadjacency matrix:
        self.adj_matrix = self.get_biadjacency_matrix(self.sol.x)
        self.print_max_degree_differences()
        # self.test_average_degrees()

# ------------------------------------------------------------------------------
# Solve coupled nonlinear equations and get BiCM biadjacency matrix
# ------------------------------------------------------------------------------

    def solve_equations(self, x0=None, method='hybr', jac=None, tol=None,
                        **kwargs):
        """Solve the system of equations of the maximum log-likelihood problem.

        The system of equations is solved using ``scipy``'s root function with
        the solver defined by ``method``. The solutions correspond to the
        Lagrange multipliers

        .. math::

            x_i = \exp(-\\theta_i).

        Depending on the solver, keyword arguments ``kwargs`` can be passed to
        the solver. Please refer to the `scipy.optimize.root documentation
        <https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/
        scipy.optimize.root.html>`_ for detailed descriptions.

        The default solver is the modified Powell method ``hybr``. Least-squares
        can be chosen with ``method='lm'`` for the Levenberg-Marquardt approach.

        .. note::

            It can happen that the solver ``method`` used by ``scipy.root``
            does not converge to a solution.
            In this case, please try another ``method`` or different initial
            conditions and refer to the `scipy.optimize.root documentation
            <https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/
            scipy.optimize.root.html>`_.

        :param x0: initial guesses for the solutions.
        :type x0: numpy.array, optional
        :param method: type of solver, default is ‘hybr’. For other
            solvers, see the `scipy.optimize.root documentation
            <https://docs.scipy.org/doc/
            scipy-0.19.0/reference/generated/scipy.optimize.root.html>`_.
        :type method: str, optional
        :param jac: Jacobian of the system
        :type jac: bool or callable, optional
        :param tol: tolerance for termination. For detailed control, use
            solver-specific options.
        :type tol: float, optional
        :param kwargs: solver-specific options, please refer to the SciPy
            documentation
        :returns: solution of the equation system
        :rtype: scipy.optimize.OptimizeResult

        :raise ValueError: raise an error if not enough initial conditions
            are provided
        """
        # use Jacobian if the hybr solver is chosen
        if method is 'hybr':
            jac = self.jacobian

        # set initial conditions
        if x0 is None:
            x0 = self.dseq / np.sqrt(np.sum(self.dseq))
        else:
            if not len(x0) == self.dim:
                msg = "One initial condition for each parameter is required."
                raise ValueError(msg)

        # solve equation system
        sol = opt.root(fun=self.equations, x0=x0, method=method, jac=jac,
                       tol=tol, **kwargs)

        # check whether system has been solved successfully
        print "Solver successful:", sol.success
        print sol.message
        if not sol.success:
            errmsg = "Try different initial conditions and/or a" + \
                     "different solver, see documentation at " + \
                     "https://docs.scipy.org/doc/scipy-0.19.0/reference/" + \
                     "generated/scipy.optimize.root.html"
            print errmsg
        return sol

    def equations(self, xx):
        """Return the equations of the log-likelihood maximization problem.

        Note that the equations for the row-nodes depend only on the
        column-nodes and vice versa, see [Saracco2015]_.

        :param xx: Lagrange multipliers which have to be solved
        :type xx: numpy.array
        :returns: equations to be solved (:math:`f(x) = 0`)
        :rtype: numpy.array
        """
        eq = -self.dseq
        for i in xrange(0, self.num_rows):
            for j in xrange(self.num_rows, self.dim):
                dum = xx[i] * xx[j] / (1. + xx[i] * xx[j])
                eq[i] += dum
                eq[j] += dum
        return eq

    def jacobian(self, xx):
        """Return a NumPy array with the Jacobian of the equation system.

        :param xx: Lagrange multipliers which have to be solved
        :type xx: numpy.array
        :returns: Jacobian
        :rtype: numpy.array
        """
        jac = np.zeros((self.dim, self.dim))
        for i in xrange(0, self.num_rows):
            # df_c / df_c' = 0 for all c' != c
            for j in xrange(self.num_rows, self.dim):
                # df_c / dx_c != 0
                xxi = xx[i] / (1.0 + xx[i] * xx[j]) ** 2
                xxj = xx[j] / (1.0 + xx[i] * xx[j]) ** 2
                jac[i, i] += xxj
                jac[i, j] = xxi
                jac[j, i] = xxj
                jac[j, j] += xxi
        return jac

    def get_biadjacency_matrix(self, xx):
        """ Calculate the biadjacency matrix of the null model.

        The biadjacency matrix describes the BiCM null model, i.e. the optimal
        average graph :math:`<G>^*` with the average link probabilities
        :math:`<G>^*_{rc} = p_{rc}` ,
        :math:`p_{rc} = \\frac{x_r \\cdot x_c}{1 + x_r\\cdot x_c}.`
        :math:`x` are the solutions of the equation system which has to be
        solved for the null model.
        Note that :math:`r` and :math:`c` are taken from opposite bipartite
        node sets, thus :math:`r \\neq c`.

        :param xx: solutions of the equation system (Lagrange multipliers)
        :type xx: numpy.array
        :returns: biadjacency matrix of the null model
        :rtype: numpy.array

        :raises ValueError: raise an error if :math:`p_{rc} < 0` or
            :math:`p_{rc} > 1` for any :math:`r, c`
        """
        mat = np.empty((self.num_rows, self.num_columns))
        xp = xx[range(self.num_rows, self.dim)]
        for i in xrange(self.num_rows):
            mat[i, ] = xx[i] * xp / (1 + xx[i] * xp)
        # account for machine precision:
        mat += np.finfo(np.float).eps
        if np.any(mat < 0):
            errmsg = 'Error in get_adjacency_matrix: probabilities < 0 in ' \
                  + str(np.where(mat < 0))
            raise ValueError(errmsg)
        elif np.any(mat > (1. + np.finfo(np.float).eps)):
            errmsg = 'Error in get_adjacency_matrix: probabilities > 1 in' \
                  + str(np.where(mat > 1))
            raise ValueError(errmsg)
        assert mat.shape == self.bin_mat.shape, \
            "Biadjacency matrix has wrong dimensions."
        return mat

# ------------------------------------------------------------------------------
# Test correctness of results:
# ------------------------------------------------------------------------------

    def print_max_degree_differences(self):
        """Print the maximal differences between input network and BiCM degrees.

        Check that the degree sequence of the solved BiCM null model graph
        corresponds to the degree sequence of the input graph.
        """
        ave_deg_columns =np.sum(self.adj_matrix, axis=0)
        ave_deg_rows = np.sum(self.adj_matrix, axis=1)
        print "Maximal degree differences between data and BiCM:"
        print "Columns:", np.abs(np.max(
            self.dseq[self.num_rows:] - ave_deg_columns))
        print "Rows:", np.abs(np.max(
            self.dseq[:self.num_rows] - ave_deg_rows))

    def test_average_degrees(self):
        """Test the constraints on the node degrees.

        Check that the degree sequence of the solved BiCM null model graph
        corresponds to the degree sequence of the input graph.
        """
        ave_deg_columns = np.squeeze(np.sum(self.adj_matrix, axis=0))
        ave_deg_rows = np.squeeze(np.sum(self.adj_matrix, axis=1))
        eps = 1e-2  # error margin
        c_derr = np.where(np.logical_or(
            # average degree too small:
            ave_deg_rows + eps < self.dseq[:self.num_rows],
            # average degree too large:
            ave_deg_rows - eps > self.dseq[:self.num_rows]))
        p_derr = np.where(np.logical_or(
            ave_deg_columns + eps < self.dseq[self.num_rows:],
            ave_deg_columns - eps > self.dseq[self.num_rows:]))
        # Check row-nodes degrees:
        if not np.array_equiv(c_derr, np.array([])):
            print '...inaccurate row-nodes degrees:'
            for i in c_derr[0]:
                print 'Row-node ', i, ':',
                print 'input:', self.dseq[i], 'average:', ave_deg_rows[i]
        # Check column-nodes degrees:
        if not np.array_equiv(p_derr, np.array([])):
            print '...inaccurate column-nodes degrees:'
            for i in c_derr[0]:
                print 'Column-node ', i, ':',
                print 'input:', self.dseq[i + self.num_rows], \
                    'average:', ave_deg_columns[i]

# ------------------------------------------------------------------------------
# Lambda motifs
# ------------------------------------------------------------------------------

    def lambda_motifs(self, bip_set, parallel=True, filename=None,
            delim='\t', binary=True, num_chunks=4):
        """Calculate and save the p-values of the :math:`\\Lambda`-motifs.

        For each node couple in the bipartite layer specified by ``bip_set``,
        calculate the p-values of the corresponding :math:`\\Lambda`-motifs
        according to the link probabilities in the biadjacency matrix of the
        BiCM null model.

        The results can be saved either as a binary ``.npy`` or a
        human-readable ``.csv`` file, depending on ``binary``.

        .. note::

            * The total number of p-values that are calculated is split into
              ``num_chunks`` chunks, which are processed sequentially in order
              to avoid memory allocation errors. Note that a larger value of
              ``num_chunks`` will lead to less memory occupation, but comes at
              the cost of slower processing speed.

            * The output consists of a one-dimensional array of p-values. If
              the bipartite layer ``bip_set`` contains ``n`` nodes, this means
              that the array will contain :math:`\\binom{n}{2}` entries. The
              indices ``(i, j)`` of the nodes corresponding to entry ``k`` in
              the array can be reconstructed using the method
              :func:`BiCM.flat2_triumat_idx`. The number of nodes ``n``
              can be recovered from the length of the array with
              :func:`BiCM.flat2_triumat_dim`

            * If ``binary == False``, the ``filename`` should end with
              ``.csv``. If ``binary == True``, it will be saved in binary NumPy
              ``.npy`` format and the suffix ``.npy`` will be appended
              automatically. By default, the file is saved in binary format.

        :param bip_set: select row-nodes (``True``) or column-nodes (``False``)
        :type bip_set: bool
        :param parallel: select whether the calculation of the p-values should
            be run in parallel (``True``) or not (``False``)
        :type parallel: bool
        :param filename: name of the output file
        :type filename: str
        :param delim: delimiter between entries in the ``.csv``file, default is
            ``\\t``
        :type delim: str
        :param binary: if ``True``, the file will be saved in the binary
            NumPy format ``.npy``, otherwise as ``.csv``
        :type binary: bool
        :param num_chunks: number of chunks of p-value calculations that are
            performed sequentially
        :type num_chunks: int
        :raise ValueError: raise an error if the parameter ``bip_set`` is
            neither ``True`` nor ``False``
        """
        if (type(bip_set) == bool) and bip_set:
            biad_mat = self.adj_matrix
            bin_mat = self.bin_mat
        elif (type(bip_set) == bool) and not bip_set:
            biad_mat = np.transpose(self.adj_matrix)
            bin_mat = np.transpose(self.bin_mat)
        else:
            errmsg = "'" + str(bip_set) + "' " + 'not supported.'
            raise NameError(errmsg)

        n = self.get_triup_dim(bip_set)
        pval = np.ones(shape=(n, ), dtype='float') * (-0.1)

        # handle layers of dimension 2 separately
        if n == 1:
            nlam = np.dot(bin_mat[0, :], bin_mat[1, :].T)
            plam = biad_mat[0, :] * biad_mat[1, :]
            pb = PoiBin(plam)
            pval[0] = pb.pval(nlam)
        else:
            # if the dimension of the network is too large, split the
            # calculations # of the p-values in ``m`` intervals to avoid memory
            # allocation errors
            if n > 100:
                kk = self.split_range(n, m=num_chunks)
            else:
                kk = [0]
            # calculate p-values for index intervals
            for i in range(len(kk) - 1):
                k1 = kk[i]
                k2 = kk[i + 1]
                nlam = self.get_lambda_motif_block(bin_mat, k1, k2)
                plam = self.get_plambda_block(biad_mat, k1, k2)
                pv = self.get_pvalues_q(plam, nlam, k1, k2)
                pval[k1:k2] = pv
            # last interval
            k1 = kk[len(kk) - 1]
            k2 = n - 1
            nlam = self.get_lambda_motif_block(bin_mat, k1, k2)
            plam = self.get_plambda_block(biad_mat, k1, k2)
            # for the last entry we have to INCLUDE k2, thus k2 + 1
            pv = self.get_pvalues_q(plam, nlam, k1, k2 + 1)
            pval[k1:] = pv
        # check that all p-values have been calculated
#        assert np.all(pval >= 0) and np.all(pval <= 1)
        if filename is None:
            fname = 'p_values_' + str(bip_set)
            if not binary:
                fname +=  '.csv'
        else:
            fname = filename
        # account for machine precision:
        pval += np.finfo(np.float).eps
        self.save_array(pval, filename=fname, delim=delim,
                         binary=binary)

    def get_lambda_motif_block(self, mm, k1, k2):
        """Return a subset of :math:`\\Lambda`-motifs as observed in ``mm``.

        Given the binary input matrix ``mm``, count the number of
        :math:`\\Lambda`-motifs for all the node couples specified by the
        interval :math:`\\left[k_1, k_2\\right[`.


        .. note::

            * The :math:`\\Lambda`-motifs are counted between the **row-nodes**
              of the input matrix ``mm``.

            * If :math:`k_2 \equiv \\binom{mm.shape[0]}{2}`, the interval
              becomes :math:`\\left[k_1, k_2\\right]`.

        :param mm: binary matrix
        :type mm: numpy.array
        :param k1: lower interval limit
        :type k1: int
        :param k2: upper interval limit
        :type k2: int
        :returns: array of observed :math:`\\Lambda`-motifs
        :rtype: numpy.array
        """
        ndim = mm.shape[0]
        # if the upper limit is the largest possible index, i.e. corresponds to
        # the node couple (ndim - 2, ndim - 1), where node indices start from 0,
        # include the result
        if k2 == (ndim * (ndim - 1) / 2 - 1):
            flag = 1
        else:
            flag = 0
        aux = np.ones(shape=(k2 - k1 + flag, )) * (-1) # -1 as a test
        [i1, j1] = self.flat2triumat_idx(k1, ndim)
        [i2, j2] = self.flat2triumat_idx(k2, ndim)

        # if limits have the same row index
        if i1 == i2:
            aux[:k2 - k1] = np.dot(mm[i1, :], mm[j1:j2, :].T)
        # if limits have different row indices
        else:
            k = 0
            # get values for lower limit row
            fi = np.dot(mm[i1, :], mm[j1:, :].T)
            aux[:len(fi)] = fi
            k += len(fi)
            # get values for intermediate rows
            for i in range(i1 + 1, i2):
                mid = np.dot(mm[i, :], mm[i + 1:, :].T)
                aux[k : k + len(mid)] = mid
                k += len(mid)
            # get values for upper limit row
            if flag == 1:
                aux[-1] = np.dot(mm[ndim - 2, :], mm[ndim - 1, :].T)
            else:
                la =  np.dot(mm[i2, :], mm[i2 + 1 : j2, :].T)
                aux[k:] = la
        return aux

    def get_plambda_block(self, biad_mat, k1, k2):
        """Return a subset of the :math:`\\Lambda` probability matrix.

        Given the biadjacency matrix ``biad_mat`` with
        :math:`\\mathbf{M}_{rc} = p_{rc}`, which describes the probabilities of
        row-node ``r`` and column-node ``c`` being linked, the method returns
        the matrix

        :math:`P(\\Lambda)_{ij} = \\left(M_{i\\alpha_1} \\cdot M_{j\\alpha_1},
        M_{i\\alpha_2} \\cdot M_{j\\alpha_2}, \\ldots\\right),`

        for all the node couples in the interval
        :math:`\\left[k_1, k_2\\right[`.  :math:`(i, j)` are two **row-nodes**
        of ``biad_mat`` and :math:`\\alpha_k` runs over the nodes in the
        opposite layer.

        .. note::

            * The probabilities are calculated between the **row-nodes** of the
              input matrix ``biad_mat``.

            * If :math:`k_2 \equiv \\binom{biad\\_mat.shape[0]}{2}`, the
              interval becomes :math:`\\left[k1, k2\\right]`.

        :param biad_mat: biadjacency matrix
        :type biad_mat: numpy.array
        :param k1: lower interval limit
        :type k1: int
        :param k2: upper interval limit
        :type k2: int
        :returns: :math:`\\Lambda`-motif probability matrix
        :rtype: numpy.array
        """
        [ndim1, ndim2] = biad_mat.shape
        # if the upper limit is the largest possible index, i.e. corresponds to
        # the node couple (ndim - 2, ndim - 1), where node indices start from 0,
        # include the result
        if k2 == (ndim1 * (ndim1 - 1) / 2 - 1):
            flag = 1
        else:
            flag = 0
        paux = np.ones(shape=(k2 - k1 + flag, ndim2), dtype='float') * (-0.1)
        [i1, j1] = self.flat2triumat_idx(k1, ndim1)
        [i2, j2] = self.flat2triumat_idx(k2, ndim1)

        # if limits have the same row index
        if i1 == i2:
            paux[:k2 - k1, :] = biad_mat[i1, ] * biad_mat[j1:j2, :]
        # if limits have different indices
        else:
            k = 0
            # get values for lower limit row
            fi = biad_mat[i1, :] * biad_mat[j1:, :]
            paux[:len(fi), :] = fi
            k += len(fi)
            # get values for intermediate rows
            for i in range(i1 + 1, i2):
                mid = biad_mat[i, :] * biad_mat[i + 1:, :]
                paux[k : k + len(mid), :] = mid
                k += len(mid)
            # get values for upper limit row
            if flag == 1:
                paux[-1, :] = biad_mat[ndim1 - 2, :] * biad_mat[ndim1 - 1, :]
            else:
                la = biad_mat[i2, :] * biad_mat[i2 + 1:j2, :]
                paux[k:, :] = la
        return paux

    def get_pvalues_q(self, plam_mat, nlam_mat, k1, k2, parallel=True):
        """Calculate the p-values of the observed :math:`\\Lambda`-motifs.

        For each number of :math:`\\Lambda`-motifs in ``nlam_mat`` for the node
        interval :math:`\\left[k1, k2\\right[`, construct the Poisson Binomial
        distribution using the corresponding
        probabilities in ``plam_mat`` and calculate the p-value.

        :param plam_mat: array containing the list of probabilities for the
            single observations of :math:`\\Lambda`-motifs
        :type plam_mat: numpy.array (square matrix)
        :param nlam_mat: array containing the observations of
            :math:`\\Lambda`-motifs
        :type nlam_mat: numpy.array (square matrix)
        :param k1: lower interval limit
        :type k1: int
        :param k2: upper interval limit
        :type k2: int
        :param parallel: if ``True``, the calculation is executed in parallel;
            if ``False``, only one process is started
        :type parallel: bool
        """
        n = len(nlam_mat)
        # the array must be sharable to be accessible by all processes
        shared_array_base = multiprocessing.Array(ctypes.c_double, n)
        pval_mat = np.frombuffer(shared_array_base.get_obj())

        # number of processes running in parallel has to be tested.
        # good guess is multiprocessing.cpu_count() +- 1
        if parallel:
            num_procs = multiprocessing.cpu_count() - 1
        elif not parallel:
            num_procs = 1
        else:
            num_procs = 1
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        p_inqueue = multiprocessing.Process(target=self.add2inqueue,
                                            args=(num_procs, plam_mat, nlam_mat,
                                                k1, k2))
        p_outqueue = multiprocessing.Process(target=self.outqueue2pval_mat,
                                             args=(num_procs, pval_mat))
        ps = [multiprocessing.Process(target=self.pval_process_worker,
                                      args=()) for i in range(num_procs)]
        # start queues
        p_inqueue.start()
        p_outqueue.start()
        # start processes
        for p in ps:
            p.start()       # each process has an id, p.pid
        p_inqueue.join()
        for p in ps:
            p.join()
        p_outqueue.join()
        return pval_mat

    def add2inqueue(self, nprocs, plam_mat, nlam_mat, k1, k2):
        """Add elements to the in-queue to calculate the p-values.

        :param nprocs: number of processes running in parallel
        :type nprocs: int
        :param plam_mat: array containing the list of probabilities for the
            single observations of :math:`\\Lambda`-motifs
        :type plam_mat: numpy.array (square matrix)
        :param nlam_mat: array containing the observations of
            :math:`\\Lambda`-motifs
        :type nlam_mat: numpy.array (square matrix)
        :param k1: lower interval limit
        :type k1: int
        :param k2: upper interval limit
        :type k2: int
        """
        n = len(plam_mat)
        # add tuples of matrix elements and indices to the input queue
        for k in xrange(k1, k2):
            self.input_queue.put((k - k1, plam_mat[k - k1, :],
                                  nlam_mat[k - k1]))

        # add as many poison pills "STOP" to the queue as there are workers
        for i in xrange(nprocs):
            self.input_queue.put("STOP")

    def pval_process_worker(self):
        """Calculate p-values and add them to the out-queue."""
        # take elements from the queue as long as the element is not "STOP"
        for tupl in iter(self.input_queue.get, "STOP"):
            pb = PoiBin(tupl[1])
            pv = pb.pval(int(tupl[2]))
            # add the result to the output queue
            self.output_queue.put((tupl[0], pv))
        # once all the elements in the input queue have been dealt with, add a
        # "STOP" to the output queue
        self.output_queue.put("STOP")

    def outqueue2pval_mat(self, nprocs, pvalmat):
        """Put the results from the out-queue into the p-value array."""
        # stop the work after having met nprocs times "STOP"
        for work in xrange(nprocs):
            for val in iter(self.output_queue.get, "STOP"):
                k = val[0]
                pvalmat[k] = val[1]

    def get_triup_dim(self, bip_set):
        """Return the number of possible node couples in ``bip_set``.

        :param bip_set: selects row-nodes (``True``) or column-nodes
            (``False``)
        :type bip_set: bool
        :returns: return the number of node couple combinations corresponding
            to the layer ``bip_set``
        :rtype: int

        :raise ValueError: raise an error if the parameter ``bip_set`` is
            neither ``True`` nor ``False``
        """
        if bip_set:
            return self.triumat2flat_dim(self.num_rows)
        elif not bip_set:
            return self.triumat2flat_dim(self.num_columns)
        else:
            errmsg = "'" + str(bip_set) + "' " + 'not supported.'
            raise NameError(errmsg)

    def split_range(self, n, m=4):
        """Split the interval :math:`\\left[0,\ldots, n\\right]` in ``m`` parts.

        :param n: upper limit of the range
        :type n: int
        :param m: number of part in which range should be split
        :type n: int
        :returns: delimiter indices for the ``m`` parts
        :rtype: list
        """
        return [i * n / m for i in range(m)]

# ------------------------------------------------------------------------------
# Auxiliary methods
# ------------------------------------------------------------------------------

    @staticmethod
    def triumat2flat_idx(i, j, n):
        """Convert an matrix index couple to a flattened array index.

        Given a square matrix of dimension ``n`` and the index couple
        ``(i, j)`` *of the upper triangular part* of the matrix, return the
        index which the matrix element would have in a flattened array.

        .. note::
            * :math:`i \\in [0, ..., n - 1]`
            * :math:`j \\in [i + 1, ..., n - 1]`
            * returned index :math:`\\in [0,\\, n (n - 1) / 2 - 1]`

        :param i: row index
        :type i: int
        :param j: column index
        :type j: int
        :param n: dimension of the square matrix
        :type n: int
        :returns: flattened array index
        :rtype: int
        """
        return int((i + 1) * n - (i + 2) * (i + 1) / 2. - (n - (j + 1)) - 1)

    @staticmethod
    def triumat2flat_dim(n):
        """Return the size of the triangular part of a ``n x n`` matrix.

        :param n: the dimension of the square matrix
        :type n: int
        :returns: number of elements in the upper triangular part of the matrix
            (excluding the diagonal)
        :rtype: int
        """
        return n * (n - 1) / 2

    @staticmethod
    def flat2triumat_dim(k):
        """Return the dimension of the matrix hosting ``k`` triangular elements.

        :param k: the number of elements in the upper triangular
            part of the corresponding square matrix, excluding the diagonal
        :type k: int
        :returns: dimension of the corresponding square matrix
        :rtype: int
        """
        return int(0.5 + np.sqrt(0.25 + 2 * k))

    @staticmethod
    def flat2triumat_idx(k, n):
        """Convert an array index into the index couple of a triangular matrix.

        ``k`` is the index of an array of length :math:`\\binom{n}{2}{2}`,
        which contains the elements of an upper triangular matrix of dimension
        ``n`` excluding the diagonal. The function returns the index couple
        :math:`(i, j)` that corresponds to the entry ``k`` of the flat array.

        .. note::
            * :math:`k \\in \left[0,\\ldots, \\binom{n}{2} - 1\\right]`
            * returned indices:
                * :math:`i \\in [0,\\ldots, n - 1]`
                * :math:`j \\in [i + 1,\\ldots, n - 1]`

        :param k: flattened array index
        :type k: int
        :param n: dimension of the square matrix
        :type n: int
        :returns: matrix index tuple (row, column)
        :rtype: tuple
        """
        # row index of array index k in the the upper triangular part of the
        # square matrix
        r = n - 2 - int(0.5 * np.sqrt(-8 * k + 4 * n * (n - 1) - 7) - 0.5)
        # column index of array index k in the the upper triangular part of the
        # square matrix
        c = k + 1 + r * (3 - 2 * n + r) / 2
        return (r, c)

    def save_biadjacency(self, filename, delim='\t', binary=False):
        """Save the biadjacendy matrix of the BiCM null model.

        The matrix can either be saved as a binary NumPy ``.npy`` file or as a
        human-readable ``.csv`` file.

        .. note::

            * The relative path has to be provided in the filename, e.g.
              *../data/pvalue_matrix.csv*.

            * If ``binary==True``, NumPy
              automatically appends the format ending ``.npy`` to the file.

        :param filename: name of the output file
        :type filename: str
        :param delim: delimiter between values in file
        :type delim: str
        :param binary: if ``True``, save as binary ``.npy``, otherwise as a
            ``.csv`` file
        :type binary: bool
        """
        self.save_array(self.adj_matrix, filename, delim, binary)

    @staticmethod
    def save_array(mat, filename, delim='\t', binary=False):
        """Save the array ``mat`` in the file ``filename``.

        The array can either be saved as a binary NumPy ``.npy`` file or as a
        human-readable ``.npy`` file.

        .. note::

            * The relative path has to be provided in the filename, e.g.
              *../data/pvalue_matrix.csv*.

            * If ``binary==True``, NumPy
              automatically appends the format ending ``.npy`` to the file.

        :param mat: array
        :type mat: numpy.array
        :param filename: name of the output file
        :type filename: str
        :param delim: delimiter between values in file
        :type delim: str
        :param binary: if ``True``, save as binary ``.npy``, otherwise as a
            ``.csv`` file
        :type binary: bool
        """
        if binary:
            np.save(filename, mat)
        else:
            np.savetxt(filename, mat, delimiter=delim)

################################################################################
# Main
################################################################################

if __name__ == "__main__":
    pass
