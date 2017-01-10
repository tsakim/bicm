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
    array as input, the module calculates the biadjacency matrix for the
    corresponding ensemble average graph :math:`<G>^*`, where the matrix
    entries correspond to the link probabilities :math:`<G>^*_{rc} = p_{rc}`
    between nodes of the two distinct bipartite node sets. Subsequently, one
    can calculate the p-values of the node similarities for nodes in the same
    bipartite layer [Saracco2015]_.

Usage:
    Be ``mat`` a two-dimensional binary NumPy array. The nodes of the two
    bipartite layers are ordered along the columns and rows, respectively. In
    the algorithm, the two layers are identified by the boolean values ``True``
    for the **row-nodes** and ``False`` for the **column-nodes**.

    Import the module and initialize the Bipartite Configuration Model::

        >>> from src.bicm import BiCM
        >>> cm = BiCM(bin_mat=mat)

    To create the biadjacency matrix of the BiCM, use::

        >>> cm.make_bicm()

    The biadjacency matrix of the BiCM null model can be saved in *<filename>*::

        >>> cm.save_matrix(cm.adj_matrix, filename=<filename>, delim='\\t')

    By default, the file is saved in a human-readable CSV format. The
    information can also be saved as a binary NumPy file ``.npy`` by using::

        >>> cm.save_matrix(cm.adj_matrix, filename=<filename>, binary=True)

    In order to analyze the similarity of the row-layer nodes and to save the
    p-values of the corresponding :math:`\Lambda`-motifs, i.e. of the number of
    shared neighbors [Saracco2016]_, use::

        >>> cm.lambda_motifs(True, filename='p_values_True.csv', delim='\\t')

    For the column-layer nodes, use::

        >>> cm.lambda_motifs(False, filename='p_values_False.csv', delim='\\t')

    Subsequently, the p-values can be used to perform a multiple hypotheses
    testing and to obtain statistically validated monopartite projections
    [Saracco2016]_. The p-values are calculated in parallel by default.

.. note::

    Since the calculation of the p-values is computationally demanding, the
    ``bicm`` module uses the Python `multiprocessing
    <https://docs.python.org/2/library/multiprocessing.html>`_ package by
    default for this purpose.  The number of parallel processes depends on the
    number of CPUs of the work station (see variable ``numprocs`` in the method
    :func:`BiCM.get_pvalues_q`).

    If the calculation should **not** be performed in parallel, use::

        >>> cm.lambda_motifs(<bool>, parallel=False)

    instead of::

        >>> cm.lambda_motifs(<bool>)

Reference:
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
    """Bipartite Configuration model for undirected binary bipartite networks.

    This class implements the Bipartite Configuration Model (BiCM), which can
    be used as a null model for the analysis of undirected and binary bipartite
    networks. The class provides methods to calculate the biadjacency matrix of
    the null model and to calculate node similarities in terms of p-values.
    """

    def __init__(self, bin_mat):
        """Initialize the parameters of the BiCM.

        :param bin_mat: binary input matrix describing the biadjacency matrix
                of a bipartite graph with the nodes of one layer along the rows
                and the nodes of the other layer along the columns.
        :type bin_mat: numpy.array
        """
        self.bin_mat = np.array(bin_mat)
        self.check_input_matrix_is_binary()
        [self.num_rows, self.num_columns] = self.bin_mat.shape
        self.dseq = self.set_degree_seq()
        self.dim = self.dseq.size
        self.sol = None             # solution of the equation system
        self.adj_matrix = None      # biadjacency matrix of the null model
        self.pval_mat = None        # matrix containing the resulting p-values
        self.input_queue = None     # queue for parallel processing
        self.output_queue = None    # queue for parallel processing

    def check_input_matrix_is_binary(self):
        """Check that the input matrix is binary, i.e. entries are 0 or 1.

        :raise AssertationError: raise an error if the input matrix is not
            binary
        """
        assert np.all(np.logical_or(self.bin_mat == 0, self.bin_mat == 1)), \
            "Input matrix is not binary."

    def set_degree_seq(self):
        """Return the node degree sequence of the input matrix.

        :returns: node degree sequence [degrees row-nodes, degrees column-nodes]
        :rtype: numpy.array
        """
        dseq = np.empty(self.num_rows + self.num_columns)
        dseq[self.num_rows:] = np.squeeze(np.sum(self.bin_mat, axis=0))
        dseq[:self.num_rows] = np.squeeze(np.sum(self.bin_mat, axis=1))
        assert dseq.size == (self.num_rows + self.num_columns)
        return dseq

    def make_bicm(self):
        """Create the biadjacency matrix of the BiCM null model.

        Solve the log-likelihood maximization problem to obtain the BiCM
        null model which respects constraints on the the degree sequence of the
        input matrix.

        :raise AssertationError: raise an error if the adjacency matrix of the
            null model has different dimensions than the input matrix
        """
        # print "+++Generating bipartite configuration model..."
        self.sol = self.solve_equations(self.equations, self.jacobian)
        # biadjacency matrix:
        self.adj_matrix = self.get_biadjacency_matrix(self.sol.x)
        # assert size of matrix
        assert self.adj_matrix.shape == self.bin_mat.shape, \
            "Biadjacency matrix has wrong dimensions."
        self.test_average_degrees()

# ------------------------------------------------------------------------------
# Solve coupled nonlinear equations and get BiCM biadjacency matrix
# ------------------------------------------------------------------------------

    def solve_equations(self, eq, jac):
        """Solve the system of equations of the maximum log-likelihood problem.

        The system of equations is solved using ``scipy``'s root function. The
        solutions correspond to the Lagrange multipliers

            :math:`x_i = \exp(-\\theta_i).`

        :param eq: system of equations (:math:`f(x) = 0`)
        :type eq: numpy.array
        :param jac: Jacobian of the system
        :type jac: numpy.ndarray
        :returns: solution of the equation system
        """
        init_guess = 0.5 * np.ones(self.dim)
        sol = opt.root(fun=eq, x0=init_guess, jac=jac)
        return sol

    def equations(self, xx):
        """Return the equations of the log-likelihood maximization problem.

        Note that the equations for the row-nodes depend only on the
        column-nodes and vice versa, see reference mentioned in the header.

        :param xx: Lagrange multipliers which have to be solved
        :type xx: numpy.array
        :returns: equations to be solved (of the form :math:`f(x) = 0`)
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
        :rtype: 2d numpy.array
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
        :math:`p_{rc} = \\frac{x_r \\cdot x_c}{1.0 + x_r\\cdot x_c}.`
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
        if np.any(mat < 0):
            errmsg = 'Error in get_adjacency_block: probabilities < 0 in ' \
                  + str(np.where(mat < 0))
            raise ValueError(errmsg)
        elif np.any(mat > 1):
            errmsg = 'Error in get_adjacency_block: probabilities > 1 in' \
                  + str(np.where(mat > 1))
            raise ValueError(errmsg)
        return mat

# ------------------------------------------------------------------------------
# Test correctness of results:
# ------------------------------------------------------------------------------

    def test_average_degrees(self):
        """Test the constraints on the node degrees.

        Assert that the degree sequence of the solved BiCM null model graph
        corresponds to the degree sequence of the input graph.
        """
        ave_deg_columns = np.squeeze(np.sum(self.adj_matrix, axis=0))
        ave_deg_rows = np.squeeze(np.sum(self.adj_matrix, axis=1))
        eps = 1./10  # error margin
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

    def lambda_motifs(self, bip_set, parallel=True, filename=None, delim='\t'):
        """Calculate and save the p-values of the :math:`\\Lambda`-motifs.

        For each node couple in the bipartite layer specified by ``bip_set``,
        :math:`\\Lambda`-motifs and calculate the corresponding p-value.

        :param bip_set: select row-nodes (``True``) or column-nodes (``False``)
        :type bip_set: bool
        :param parallel: select whether the calculation of the p-values should
            be run in parallel (``True``) or not (``False``)
        :type parallel: bool
        :param filename: name of the file which will contain the p-values
        :param delim: delimiter between entries in file, default is tab
        """
        plam_mat = self.get_plambda_matrix(self.adj_matrix, bip_set)
        nlam_mat = self.get_lambda_motif_matrix(self.bin_mat, bip_set)
        self.get_pvalues_q(plam_mat, nlam_mat, parallel)
        if filename is None:
            fname = 'p_values_' + str(bip_set) + '.csv'
        else:
            fname = filename
        self.save_matrix(self.pval_mat, filename=fname, delim=delim)

    @staticmethod
    def get_plambda_matrix(biad_mat, bip_set):
        """Return the :math:`\\Lambda`-motif probability tensor for ``bip_set``.

        Given the biadjacency matrix ``biad_mat``,
        :math:`\\mathbf{M}_{rc} = p_{rc}`, which contains the probabilities of
        row-node `r` and column-node `c` being linked, the method returns the
        tensor

        :math:`P(\\Lambda)_{ij} = (M_{i\\alpha_1} \\cdot M_{j\\alpha_1},
        M_{i\\alpha_2} \\cdot M_{j\\alpha_2}, ...),`

        where :math:`(i, j)` are two nodes of the bipartite layer ``bip_set``
        and :math:`\\alpha_k` runs over the nodes in the opposite layer.

        :param biad_mat: biadjacency matrix
        :type biad_mat: numpy.array
        :param bip_set: selects row-nodes (``True``) or column-nodes (``False``)
        :type bip_set: bool
        """
        if (type(bip_set) == bool) and bip_set:
            mm_mat = biad_mat
        elif (type(bip_set) == bool) and not bip_set:
            mm_mat = np.transpose(biad_mat)
        else:
            errmsg = "'" + str(bip_set) + "' " + 'not supported.'
            raise NameError(errmsg)
        pl2 = np.empty((mm_mat.shape[0], mm_mat.shape[0], mm_mat.shape[1]))
        for i in xrange(mm_mat.shape[0]):
            pl2[i, ] = mm_mat[i, ] * mm_mat
        di = np.diag_indices(pl2.shape[0], 2)       # set diagonal to zero
        pl2[di] = 0
        return pl2

    @staticmethod
    def get_lambda_motif_matrix(mm, bip_set):
        """Return the number of :math:`\\Lambda`-motifs as found in ``mm``.

        Given the binary input matrix ``mm``, count the number of
        :math:`\\Lambda`-motifs between node couples of the bipartite layer
        specified by ``bip_set``.

        :param mm: binary matrix
        :type mm: numpy.array
        :param bip_set: selects row-nodes (``True``) or column-nodes (``False``)
        :type bip_set: bool

        :returns: square matrix of observed :math:`\\Lambda`-motifs
        :rtype: numpy.array
        """
        if (type(bip_set) == bool) and bip_set:
            nlam_mat = np.dot(mm, np.transpose(mm))
        elif (type(bip_set) == bool) and not bip_set:
            nlam_mat = np.dot(np.transpose(mm), mm)
        else:
            errmsg = "'" + str(bip_set) + "' " + 'not supported.'
            raise NameError(errmsg)
        di = np.diag_indices(nlam_mat.shape[0], 2)
        nlam_mat[di] = 0
        return nlam_mat

    def get_pvalues_q(self, plam_mat, nlam_mat, parallel=True):
        """Calculate the p-values of the observed :math:`\\Lambda`-motifs.

        For each number of :math:`\\Lambda`-motifs in ``nlam_mat``,
        construct the Poisson Binomial distribution using the corresponding
        probabilities in ``plam_mat`` and calculate the p-value.

        .. note::
            the lower-triangular part of the output matrix is null since
            the matrix is symmetric by definition.

        :param plam_mat: array containing the list of probabilities for the
                        single observations of :math:`\\Lambda`-motifs
        :type plam_mat: numpy.array (square matrix)
        :param nlam_mat: array containing the observations of
            :math:`\\Lambda`-motifs
        :type nlam_mat: numpy.array (square matrix)
        :param parallel: if ``True``, the calculation is executed in parallel;
                        if ``False``, only one process is started
        :type parallel: bool
        :return pval_mat: array containing the p-values corresponding to the
                        :math:`\\Lambda`-values in ``nlam_mat``
        :rtype: numpy.array
        """
        n = nlam_mat.shape[0]
        # the array must be sharable to be accessible by all processes
        shared_array_base = multiprocessing.Array(ctypes.c_double, n * n)
        pval_mat = np.frombuffer(shared_array_base.get_obj())
        self.pval_mat = pval_mat.reshape(n, n)
        # number of processes running in parallel has to be tested.
        # good guess is multiprocessing.cpu_count() +- 1
        if parallel:
            numprocs = multiprocessing.cpu_count() - 1
        elif not parallel:
            numprocs = 1
        else:
            numprocs = 1
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        p_inqueue = multiprocessing.Process(target=self.add2inqueue,
                                            args=(numprocs, plam_mat, nlam_mat))
        p_outqueue = multiprocessing.Process(target=self.outqueue2pval_mat,
                                             args=(numprocs, ))
        ps = [multiprocessing.Process(target=self.pval_process_worker,
                                      args=()) for i in range(numprocs)]
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

    def add2inqueue(self, nprocs, plam_mat, nlam_mat):
        """Add elements to the in-queue to calculate the p-values.

        :param nprocs: number of processes running in parallel
        :type nprocs: int
        :param plam_mat: array containing the list of probabilities for the
            single observations of :math:`\\Lambda`-motifs
        :type plam_mat: numpy.array (square matrix)
        :param nlam_mat: array containing the observations of
            :math:`\\Lambda`-motifs
        :type nlam_mat: numpy.array (square matrix)
        """
        n = plam_mat.shape[0]
        # add tuples of matrix elements and indices to the input queue
        for i in xrange(n):
            for j in xrange(i + 1, n):
                self.input_queue.put((i, j, plam_mat[i, j], nlam_mat[i, j]))
        # add as many poison pills "STOP" to the queue as there are workers
        for i in xrange(nprocs):
                self.input_queue.put("STOP")

    def pval_process_worker(self):
        """Calculate one p-value and add the result to the out-queue."""
        # take elements from the queue as long as the element is not "STOP"
        for tupl in iter(self.input_queue.get, "STOP"):
            pb = PoiBin(tupl[2])
            pv = pb.pval(int(tupl[3]))
            # add the result to the output queue
            self.output_queue.put((tupl[0], tupl[1], pv))
        # once all the elements in the input queue have been dealt with, add a
        # "STOP" to the output queue
        self.output_queue.put("STOP")

    def outqueue2pval_mat(self, nprocs):
        """Put the results from the out-queue into the p-value matrix."""
        # stop the work after having met nprocs times "STOP"
        for work in xrange(nprocs):
            for val in iter(self.output_queue.get, "STOP"):
                i = val[0]
                j = val[1]
                self.pval_mat[i, j] = val[2]

# ------------------------------------------------------------------------------
# Probability distributions for Lambda values
# ------------------------------------------------------------------------------
#
#    def save_lambda_probdist(self, bip_set, parallel=True, write=True,
#                             filename=None, delim='\t', binary=True):
#        """Calculate the probabilities of all :math:`\\Lambda`-motif values.
#
#        For each node pair :math:`(i, j)` in ``bip_set``, calculate the
#        probability of observing any possible number of :math:`\\Lambda`-motifs.
#        The probability matrix can either be saved as a binary NumPy ``.npy``
#        file or as a human-readable CSV file.
#
#        :param bip_set: selects row-nodes (``True``) or column-nodes (``False``)
#        :type bip_set: bool
#        :param parallel: if ``True``, the calculation is executed in parallel;
#                        if ``False``, only one process is started
#        :type parallel: bool
#        :param write: if ``True`` save the file to disk
#        :type write: bool
#        :param filename: filename. If binary is true, it should end with '.npy',
#                        otherwise with '.csv'
#        :type delim: str
#        :param delim: delimiter between values in file
#        :type delim: str
#        :param binary: if ``True``, save as binary ``.npy``,
#                     otherwise as CSV a file
#        :type binary: bool
#        :return pval_mat: array containing the p-values corresponding to the
#                        :math:`\\Lambda`-values in ``nlam_mat``
#        :rtype: numpy.array
#        """
#        plam_mat = self.get_plambda_matrix(self.adj_matrix, bip_set)
#        self.get_lambda_probdist_q(plam_mat, bip_set, parallel=parallel)
#        if write:
#            if filename is None:
#                fname = 'bicm_lambda_probdist_layer_' + str(bip_set)
#                if binary:
#                    fname += '.npy'
#                else:
#                    fname += '.csv'
#            else:
#                fname = filename
#            self.save_matrix(self.probdist_mat, filename=fname, delim=delim,
#                             binary=binary)
#
#    def get_lambda_probdist_q(self, plam_mat, bip_set, parallel=True):
#        """Apply the Poisson Binomial distribution of each node couple on
#        all the possible number of nearest neighbors they can have, i.e.
#        the set [0, 1, ..., M], where M is the number of nodes in the opposite
#        bipartite layer.
#        """
#        if bip_set:
#            n = self.num_rows
#            m = self.num_columns
#        elif not bip_set:
#            n = self.num_columns
#            m = self.num_rows
#        else:
#            errmsg = "'" + str(bip_set) + "' " + 'not supported.'
#            raise NameError(errmsg)
#
#        # the array must be sharable to be accessible by all processes
#        shared_array_base = multiprocessing.Array(
#                            ctypes.c_double, n * (n - 1) * (m + 1) / 2)
#        probdist_mat = np.frombuffer(shared_array_base.get_obj())
#        self.probdist_mat = probdist_mat.reshape(n * (n - 1) / 2, m + 1)
#        lambda_values = np.arange(m + 1)
#
#        # number of processes running in parallel has to be tested.
#        # good guess is multiprocessing.cpu_count() +- 1
#        if parallel:
#            numprocs = multiprocessing.cpu_count() - 1
#        else:
#            numprocs = 1
#        self.input_queue = multiprocessing.Queue()
#        self.output_queue = multiprocessing.Queue()
#
#        p_inqueue = multiprocessing.Process(target=self.probdist_add2inqueue,
#                                            args=(numprocs, plam_mat,
#                                                  lambda_values))
#        p_outqueue = multiprocessing.Process(target=self.probdist_outqueue2mat,
#                                             args=(numprocs, n))
#        ps = [multiprocessing.Process(target=self.probdist_process_worker,
#                                      args=()) for i in range(numprocs)]
#        # start queues
#        p_inqueue.start()
#        p_outqueue.start()
#        # start processes
#        for p in ps:
#            p.start()       # each process has an id, p.pid
#        p_inqueue.join()
#        for p in ps:
#            p.join()
#        p_outqueue.join()
#
#    def probdist_add2inqueue(self, nprocs, plam_mat, lambda_values):
#        """Add entries to in-queue to calculate the probablity distibutions.
#
#        """
#        n = plam_mat.shape[0]
#        # add tuples of matrix elements and indices to the input queue
#        for i in xrange(n):
#            for j in xrange(i + 1, n):
#                self.input_queue.put((i, j, plam_mat[i, j], lambda_values))
#        # add as many poison pills "STOP" to the queue as there are workers
#        for i in xrange(nprocs):
#            self.input_queue.put("STOP")
#
#    def probdist_process_worker(self):
#        """Take an element from the queue and calculate the probability of the
#         possible lambda values. Add the result to the out-queue.
#        """
#        # take elements from the queue as long as the element is not "STOP"
#        for tupl in iter(self.input_queue.get, "STOP"):
#            pb = PoiBin(tupl[2])
#            lambdaprobs = pb.pmf(tupl[3])
#            # add the result to the output queue
#            self.output_queue.put((tupl[0], tupl[1], lambdaprobs))
#        # once all the elements in the input queue have been dealt with, add a
#        # "STOP" to the output queue
#        self.output_queue.put("STOP")
#
#    def probdist_outqueue2mat(self, nprocs, mat_dim):
#        """Take the results from the out-queue and put them into the lambda
#        probability matrix.
#        """
#        # stop the work after having met nprocs times "STOP"
#        for work in xrange(nprocs):
#            for val in iter(self.output_queue.get, "STOP"):
#                i = val[0]
#                j = val[1]
#                k = self.triumat2flat_idx(i, j, mat_dim)
#                self.probdist_mat[k, :] = val[2]

# ------------------------------------------------------------------------------
# Auxiliary methods
# ------------------------------------------------------------------------------

    @staticmethod
    def triumat2flat_idx(i, j, n):
        """Convert an matrix index couple to a flattened array index.

        Given a square matrix of dimension :math:`n` and an index couple
        :math:`(i, j)` *of the upper triangular part* of the matrix, the
        function returns the index which the matrix element would have in a
        flattened array.

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

    def save_biadjacency(self, filename, delim='\t', binary=False):
        """Save the biadjacendy matrix of the BiCM null model.

        The matrix can either be saved as a binary NumPy ``.npy`` file or as a
        human-readable CSV file.

        .. note:: The relative path has to be provided in the filname, e.g.
                *../data/biadjacency_matrix.csv*

        :param filename: name of the output file
        :type filename: str
        :param delim: delimiter between values in file
        :type delim: str
        :param binary: if ``True``, save as binary ``.npy``,
                     otherwise as CSV a file
        :type binary: bool
        """
        self.save_matrix(self.adj_matrix, filename, delim, binary)

    @staticmethod
    def save_matrix(mat, filename, delim='\t', binary=False):
        """Save the matrix ``mat`` in the file ``filename``.

        The matrix can either be saved as a binary NumPy ``.npy`` file or as a
        human-readable CSV file.

        .. note:: The relative path has to be provided in the filname, e.g.
                *../data/pvalue_matrix.csv*

        :param mat: two-dimensional matrix
        :type mat: numpy.array
        :param filename: name of the output file
        :type filename: str
        :param delim: delimiter between values in file
        :type delim: str
        :param binary: if ``True``, save as binary ``.npy``,
                     otherwise as CSV a file
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
