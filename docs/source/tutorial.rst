.. _tutorial:

Tutorial
========

The tutorial will take you step by step from the biadjacency matrix of a
real-data network to the calculation of the p-values. Our example bipartite
network will be the following:

.. image:: figures/nw.png
    :width: 25 %
    :align: center

The structure of the network can be caught in the `biadjacency matrix
<https://en.wikipedia.org/w/index.php?title=Adjacency_matrix&oldid=751840428#Adjacency_matrix_of_a_bipartite_graph>`_.
In our case, the matrix is 

.. math::
    \left[
    \begin{matrix}
        1 & 1 & 0 & 0 \\
        0 & 1 & 1 & 1 \\
        0 & 1 & 0 & 1 
    \end{matrix}
    \right]

Note that the nodes of the layers of the bipartite network are ordered along
the rows and the columns, respectively. In the algorithms, the two layers are
identified by the boolean values ``True`` for the **row-nodes** and ``False`` for
the **column-nodes**. In our example image, the row-nodes are colored in blue
(top layer) and the column-nodes in red (bottom layer).

Let's get started by importing the necessary modules::

    >>> import numpy
    >>> from src.bicm import BiCM

The biadjacency matrix of our toy network will be saved in the two-dimensional
NumPy array ``mat``::

    >>> mat = np.array([[1, 1, 0, 0], [0, 1, 1, 1], [0, 1, 0, 1]])

and we initialize the Bipartite Configuration Model with::

    >>> cm = BiCM(bin_mat=mat)

In order to obtain the biadjacency matrix of the BiCM null model corresponding
to the input network, a number of equations have to be solved. However, this is
done automatically by running::

    >>> cm.make_bicm()

You can now save the bidajacency matrix in the file *<filename>* as::

    >>> m.save_biadjacency(filename=<filename>, delim='\t')

Note that the default delimiter is ``\t``. Other delimiters such as ``,`` or
``;`` work fine as well. The matrix can either be saved as a human-readable
``.csv`` or as a binary NumPy ``.npy`` file, see :func:`save_biadjacency` in
the :ref:`api`. In our example graph, the BiCM matrix should be::

    >>> cm.adj_matrix 
    array([[ 0.21602144,  0.99855239,  0.21602144,  0.56873952],
           [ 0.56845256,  0.99969684,  0.56845256,  0.86309703],
           [ 0.21602144,  0.99855239,  0.21602144,  0.56873952]])

Each entry in the matrix corresponds to the probability of observing a link
between the corresponding row- and column-nodes. If we take two nodes in the
same layer, we can count the number of common neighbors that they share in the
original input network and calculate the probability of observing the same of
more common neighbors according to the BiCM [Saracco2016]_. This corresponds to
calculating the p-values for a right-sided hypothesis testing. 

The calculation of the p-values is computation and memory intensive and should
be performed in parallel, see :ref:`parallel` for details. It can be executed
by simply running::

    >>> cm.lambda_motifs(<bool>, filename=<filename>, delim='\t')

where ``<bool>`` is either ``True`` of ``False`` depending on whether one wants
to address the similarities of the **row-** or **column-nodes**, respectively,
and ``<filename>`` is the name of the output file.

.. add comment on binary/ not binary

Having calculated the p-values, it is possible to perform a multiple hypothesis
testing with FDR control and to obtain an unbiased monopartite projection of
the original bipartite network. In the projection, only statistically
significant edges are kept. 

For further information on the post-processing and the monopartite projections,
please refer to [Saracco2016]_.


