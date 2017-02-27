BiCM Quickstart
===============

If you want to get started right away, go ahead and follow the summary below.  The ``bicm`` module encompasses essentially two steps for the analysis of node similarities in bipartite networks:

#. given an input matrix, create the biadjacency matrix of the BiCM null model
#. perform a statistical validation of the similarities of nodes in the same
   layer

The validated node similarities can be used to obtain an unbiased monopartite projection of the bipartite network, as illustrated in [Saracco2016]_.

For more detailed explanations of the methods, please refer to [Saracco2016]_, the :ref:`tutorial` and the :ref:`api`.

Obtaining the biadjacency matrix of the BiCM null model
--------------------------------------------------------------------------------

Be ``mat`` a two-dimensional binary NumPy array, which describes the
`biadjacency matrix
<https://en.wikipedia.org/w/index.php?title=Adjacency_matrix&oldid=751840428#Adjacency_matrix_of_a_bipartite_graph>`_
of an undirected bipartite network. The nodes of the two bipartite layers are
ordered along the columns and rows, respectively. In the algorithm, the two
layers are identified by the boolean values ``True`` for the **row-nodes** and
``False`` for the **column-nodes**.

Import the module and initialize the Bipartite Configuration Model::

    >>> from src.bicm import BiCM
    >>> cm = BiCM(bin_mat=mat)

To create the biadjacency matrix of the BiCM, use::

    >>> cm.make_bicm()

The biadjacency matrix of the BiCM null model can be saved in *<filename>*::

    >>> cm.save_biadjacency(filename=<filename>, delim='\t')

By default, the file is saved in a human-readable CSV format. The information can also be saved as a binary NumPy file ``.npy`` by using::

    >>> cm.save_biadjacency(filename=<filename>, binary=True)

If the file is not binary, it should end with, e.g., ``.csv``. If it is binary instead, NumPy automatically attaches the ending ``.npy``.

Calculating the p-values of the node similarities
--------------------------------------------------------------------------------

In order to analyze the similarity of the row-layer nodes and to save the
p-values of the corresponding :math:`\Lambda`-motifs in *<filename>*, i.e. of the number of
shared neighbors [Saracco2016]_, use::

    >>> cm.lambda_motifs(True, filename=<filename>)
  
By default, the file is saved as binary NumPy file to reduce the required space
of the file. The format suffix ``.npy`` is appended automatically to the file
name. If the file should be saved in a human-readable CSV format, use::

    >>> cm.lambda_motifs(True, filename=<filename>, delim='\t')

For the column-layer nodes, use::

    >>> cm.lambda_motifs(False, filename='p_values_False.csv', delim='\t')

Subsequently, the p-values can be used to perform a multiple hypotheses testing
and to obtain statistically validated monopartite projections [Saracco2016]_.
The p-values are calculated in parallel by default, see :ref:`parallel` for
details.

