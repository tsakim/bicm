# -*- coding: utf-8 -*-
"""
Created on Tue Dec 1 08:04:28 2015

Module:
    bicm - Bipartite Configuration Model

Author:
    Mika Straka

Description:
    Implementation of the Bipartite Configuration Model (BiCM) for binary
    undirected bipartite networks, see reference below.

    Given a biadjacency matrix of a bipartite graph in the form of a binary
    array with rectangular shape as input, the module calulates the biadjacency
    matrix for the corresponding ensemble average graph <G>^*, where the matrix
    entries correspond to the link probabilities <G>^*_ij = p_ij between nodes
    of the two distinct bipartite node sets.

Usage:
    Be <td> a binary matrix in the form of an numpy array. The nodes of the two
    distinct bipartite layers are ordered along the columns and rows. In the
    following, rows and columns are referred to by the booleans "True" and
    "False" and the nodes are called "row-nodes" and "column-nodes",
    respectively.

    Initialize the Bipartite Configuration Model for the matrix <td> with
        $ cm = BiCM(bin_mat=td)

    To create the BiCM by solving the corresponding equation system, use
        $ cm.make_bicm()

    To get the Lambda motifs and save the corresponding p-values for the
    row-layer nodes in the folder "output", use
        $ cm.lambda_motifs(True, filename='p_values_True.csv', delim='\t')

    To get the Lambda motifs and save the corresponding p-values for the
    column-layer nodes in the folder "output", use
        $ cm.lambda_motifs(False, filename='p_values_False.csv', delim='\t')

    Note that the saving of the files requires the name of the main folder
    which contains the folder "src", which itself contains the file bicm.py.
    If the folder name is NOT the default "bicm", the function
    self.get_main_dir() has to be called as
        self.get_main_dir(main_dir_name=<main folder name>)
    in __init__(bin_mat).

Parallel computation:
    The module uses the Python multiprocessing package and the calculation of
    the p-values is executed in parallel. The number of parallel processes
    depends on the number of CPUs of the work station, see variable "numprocs"
    in method "self.get_pvalues_q".
    If the calculation should not be performed in parallel, use

        $ cm.lambda_motifs(True, parallel=False)

    and respectively

        $ cm.lambda_motifs(False, parallel=False)

Reference:
    Saracco, F. et al. Randomizing bipartite networks: the case of the World
    Trade Web.
    Sci. Rep. 5, 10595;
    doi: 10.1038/srep10595 (2015).
"""

import ctypes
import multiprocessing
import os
import scipy.optimize as opt
import numpy as np
from poibin.poibin import PoiBin
#import src.aic_bic as aic


class BiCM:
    """Create the Bipartite Configuration Model for the input matrix and
    analyze the Lambda motifs.
    """

    def __init__(self, bin_mat):
        """Initialize the parameters of the BiCM.

        :param bin_mat: binary input matrix describing the biadjacency matrix
                of a bipartite graph with the nodes of one layer along the rows
                and the nodes of the other layer along the columns.
        :type bin_mat: np.array
        """
        self.bin_mat = np.array(bin_mat)
        self.check_input_matrix_is_binary()
        [self.num_countries, self.num_products] = self.bin_mat.shape
        self.dseq = self.set_degree_seq()
        self.dim = self.dseq.size
        self.sol = None             # solution of the equation system
        self.adj_matrix = None      # biadjacency matrix of the null model
        self.pval_mat = None        # matrix containing the resulting p-values
        self.input_queue = None     # queue for parallel processing
        self.output_queue = None    # queue for parallel processing
        self.main_dir = self.get_main_dir()

    def check_input_matrix_is_binary(self):
        """Check that the input matrix is binary, i.e. entries are either
        0 or 1.
        """
        assert np.all(np.logical_or(self.bin_mat == 0, self.bin_mat == 1)), \
            "Input matrix is not binary."

    def set_degree_seq(self):
        """Set the degree sequence [degrees row-nodes, degrees column-nodes].
        """
        dseq = np.empty(self.num_countries + self.num_products)
        dseq[self.num_countries:] = np.squeeze(np.sum(self.bin_mat, axis=0))
        dseq[:self.num_countries] = np.squeeze(np.sum(self.bin_mat, axis=1))
        assert dseq.size == (self.num_countries + self.num_products)
        return dseq

    def make_bicm(self):
        """Create the biadjacency matrix of the BiCM corresponding to the
        input matrix.
        """
        # print "+++Generating bipartite configuration model..."
        self.sol = self.solve_equations(self.equations, self.jacobian)
        # biadjacency matrix:
        self.adj_matrix = self.get_biadjacency_matrix(self.sol.x)
        # assert size of matrix
        assert self.adj_matrix.shape == self.bin_mat.shape, \
            "Biadjacency matrix has wrong dimension."
        self.test_average_degrees()

# ------------------------------------------------------------------------------
# Solve system of coupled nonlinear equations &
# get biadjacency matrix
# ------------------------------------------------------------------------------

    def solve_equations(self, eq, jac):
        """Solve the system of nonlinear equations using scipy's root function.
        The solutions correspond to the Lagrange multipliers
            x_i = exp(-\theta_i).

        :param eq: system of equations f(x) = 0
        :type eq: np.array
        :param jac: Jacobian of the system
        :type jac: np.array
        """
        init_guess = 0.5 * np.ones(self.dim)
        sol = opt.root(fun=eq, x0=init_guess, jac=jac)
        return sol

    def equations(self, xx):
        """Return an array with the equations of the system. Note that the
        equations for the row-nodes depend only on the x-values of the
        column-nodes and vice versa.

        :param xx: Lagrange multipliers which have to be solved
        :type xx: np.array
        :return : np.array of equations = 0
        """
        eq = -self.dseq
        for i in xrange(0, self.num_countries):
            for j in xrange(self.num_countries, self.dim):
                dum = xx[i] * xx[j] / (1. + xx[i] * xx[j])
                eq[i] += dum
                eq[j] += dum
        return eq

    def jacobian(self, xx):
        """Return a numpy array with the Jacobian for the system of equations
        defined in self.equations.

        :param xx: Lagrange multipliers which have to be solved
        :type xx: np.array
        :return : np.ndarray Jacobian
        """
        jac = np.zeros((self.dim, self.dim))
        for i in xrange(0, self.num_countries):
            # df_c / df_c' = 0 for all c' != c
            for j in xrange(self.num_countries, self.dim):
                # df_c / dx_c != 0
                xxi = xx[i] / (1.0 + xx[i] * xx[j]) ** 2
                xxj = xx[j] / (1.0 + xx[i] * xx[j]) ** 2
                jac[i, i] += xxj
                jac[i, j] = xxi
                jac[j, i] = xxj
                jac[j, j] += xxi
        return jac

    def get_biadjacency_matrix(self, xx):
        """ Calculate the biadjacency matrix of the solved system. The
        biadjacency matrix describes the optimal average graph <G>^* with the
        elements G_ij == average link probabilities
            p_ij, p_ij = x_i * x_j / (1.0 + x_i * x_j), 
        where x are the solutions of the equations solved above.
        Note that i and j are taken from opposite bipartite node sets and
        i != j.

        :param xx: Lagrande multipliers / solutions of the system
        :type xx: np.array
        """
        mat = np.empty((self.num_countries, self.num_products))
        xp = xx[range(self.num_countries, self.dim)]
        for i in xrange(self.num_countries):
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
        """Test the degree sequence of the graph defined by the biadjacency
        matrix. The node degrees should be equal to the input sequence
        self.dseq.
        """
        ave_deg_products = np.squeeze(np.sum(self.adj_matrix, axis=0))
        ave_deg_countries = np.squeeze(np.sum(self.adj_matrix, axis=1))
        eps = 1./10  # error margin
        # print '+++Test degrees of solution...'
        c_derr = np.where(np.logical_or(
            # average degree too small:
            ave_deg_countries + eps < self.dseq[:self.num_countries],
            # average degree too big:
            ave_deg_countries - eps > self.dseq[:self.num_countries]))
        p_derr = np.where(np.logical_or(
            ave_deg_products + eps < self.dseq[self.num_countries:],
            ave_deg_products - eps > self.dseq[self.num_countries:]))
        # Check row-nodes degrees:
        if not np.array_equiv(c_derr, np.array([])):
            print '...inaccurate row-nodes degrees:'
            for i in c_derr[0]:
                print 'Country', i, ':',
                print 'input:', self.dseq[i], 'average:', ave_deg_countries[i]
        # Check column-nodes degrees:
        if not np.array_equiv(p_derr, np.array([])):
            print '...inaccurate column-nodes degrees:'
            for i in c_derr[0]:
                print 'Product', i, ':',
                print 'input:', self.dseq[i + self.num_countries], \
                    'average:', ave_deg_products[i]
        # print 'Done.'

# ------------------------------------------------------------------------------
# Lambda motifs
# ------------------------------------------------------------------------------

    def lambda_motifs(self, bip_set, parallel=True, filename=None, delim='\t'):
        """Obtain and save the p-values of the Lambda motifs observed in the
        binary inpu matrix for the node set defined by bip_set.

        :param bip_set: selects countries (True, fix rows) or
                        products (False, fix columns)
        :type bip_set: bool
        :param parallel: defines whether function should be run in parallel
                        True = use parallel processing,
                        False = don't use parallel processing
        :type parallel: bool
        """
        pl2_mat = self.get_plambda_matrix(self.adj_matrix, bip_set)
        nl2_mat = self.get_lambda_motif_matrix(self.bin_mat, bip_set)
        self.get_pvalues_q(pl2_mat, nl2_mat, parallel)
        if filename is None:
            fname = 'p_values_' + str(bip_set) + '.csv'
        self.save_pvalues(self.pval_mat, filename=fname, delim=delim)

    @staticmethod
    def get_plambda_matrix(biad_mat, bip_set):
        """Return the tensor P_{\Lambda} = P_ij^c, where i, j \in same
        bipartite node set, i.e. both rows or column indices, and with input M
        P_ij = [M_{ic_1} * M_{jc_1}, M_{ic_2} * M_{jc_2}, ...]
        over all c in the opposite bipartite node set.
        The input matrix M / biad_mat is a biadjacency matrix and bip_set
        defines which bipartite node set should be considered.

        :param biad_mat: biadjacency matrix
        :type biad_mat: np.array
        :param bip_set: selects countries (True, fix rows) or
                        products (False, fix columns)
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
        # set diagonal to zero
        di = np.diag_indices(pl2.shape[0], 2)
        pl2[di] = 0
        return pl2

    @staticmethod
    def get_lambda_motif_matrix(mm, bip_set):
        """Return the matrix of Lambda motifs based on the binary input
        matrix.

        :param mm: binary matrix (rectangular / square shaped)
        :type mm: np.array
        :param bip_set: selects countries (True, fix rows) or
                        products (False, fix columns)
        :type bip_set: bool

        :return nl2_mat: square matrix of observed Lambda motifs
        """
        if (type(bip_set) == bool) and bip_set:
            nl2_mat = np.dot(mm, np.transpose(mm))
        elif (type(bip_set) == bool) and not bip_set:
            nl2_mat = np.dot(np.transpose(mm), mm)
        else:
            errmsg = "'" + str(bip_set) + "' " + 'not supported.'
            raise NameError(errmsg)
        di = np.diag_indices(nl2_mat.shape[0], 2)
        nl2_mat[di] = 0
        return nl2_mat

    def get_pvalues_q(self, pl2_mat, nl2_mat, parallel=True):
        """Apply the Poisson Binomial distribution on the values given in
        nl2_mat using the probabilities in pl2_mat.

        :param pl2_mat: array containing the list of probabilities for the
                        single observations of Lambda motifs
        :type pl2_mat: np.array (square shaped, shape[0] == shape[1])
        :param nl2_mat: array containing the observations of Lambda motivs
        :type nl2_mat: np.array (square shaped, shape[0] == shape[1])
        :param parallel: if True, the calculation is executed in parallel. If
                            False, only one process is started.
        :type parallel: bool
        :return pval_mat: np.array containing the p-values corresponding to the
                            values in nl2_mat.

        NB: only upper triangular part of output is  != 0 since the matrix is
            symmetric by definition.
        """
        n = nl2_mat.shape[0]
        # sharable array:
        shared_array_base = multiprocessing.Array(ctypes.c_double, n * n)
        pval_mat = np.frombuffer(shared_array_base.get_obj())
        self.pval_mat = pval_mat.reshape(n, n)

        # metti qua il numero di core multiprocessing.cpu_count()
        # oppure +- qualche numero. deve essere testato
        if parallel:
            numprocs = multiprocessing.cpu_count() - 1
        elif not parallel:
            numprocs = 1
        else:
            numprocs = 1
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        p_inqueue = multiprocessing.Process(target=self.add2inqueue,
                                            args=(numprocs, pl2_mat, nl2_mat))
        p_outqueue = multiprocessing.Process(target=self.outqueue2pval_mat,
                                             args=(numprocs, ))
        ps = [multiprocessing.Process(target=self.pval_process_worker,
                                      args=()) for i in range(numprocs)]
        p_inqueue.start()
        p_outqueue.start()

        # print '...starting processes:'
        for p in ps:
            p.start()
            # print '......PID:', p.pid
        # print 'Processes started.'

        p_inqueue.join()
        for p in ps:
            p.join()
        p_outqueue.join()
        # print 'Processes done.'

    def add2inqueue(self, nprocs, pl2_mat, nl2_mat):
        """Add matrix entries to in-queue in order to calculate p-values."""
        n = pl2_mat.shape[0]
        # add tuples of matrix elements and indices to the input queue
        for i in xrange(n):
            for j in xrange(i + 1, n):
                self.input_queue.put((i, j, pl2_mat[i, j], nl2_mat[i, j]))
        # add as many poison pills "STOP" to the queue as there are workers
        for i in xrange(nprocs):
                self.input_queue.put("STOP")

    def pval_process_worker(self):
        """Take an element from the queue and calculate the p-value. Add the
        result to the out-queue.
        """
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
        """Take the results from the out-queue and put them into the p-value
        matrix.
        """
        # stop the work after having met nprocs times "STOP"
        for work in xrange(nprocs):
            for val in iter(self.output_queue.get, "STOP"):
                i = val[0]
                j = val[1]
                self.pval_mat[i, j] = val[2]

# ------------------------------------------------------------------------------
# Auxiliary methods
# ------------------------------------------------------------------------------

    @staticmethod
    def get_main_dir(main_dir_name='bicm'):
        """Return the absolute path to the main directory which contains the
        folders "src" and "output".
        Note that the default directory name is "bicm".

        :param main_dir_name: name of the main directory of the program.
        :type main_dir_name: string
        """
        s = os.getcwd()
        dirpath = s[:s.index(main_dir_name) + len(main_dir_name) + 1]
        return dirpath

    def save_pvalues(self, pval_mat, filename, delim='\t'):
        """Save the p-values matrix in a csv-file.

        :param pval_mat: matrix containing the p-values
        :param filename: name of the file for the p-value
        :param delim: delimiter between values in file.
        """
        fname = ''.join([self.main_dir, '/output/', filename])
        np.savetxt(fname, pval_mat, delimiter=delim)

################################################################################
# Main
################################################################################

if __name__ == "__main__":
    pass