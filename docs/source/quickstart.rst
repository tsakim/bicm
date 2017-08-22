BiCM Quickstart
===============

The ``bicm`` module encompasses essentially two steps for the validation of node similarities in bipartite networks:

#. Given a binary input matrix, create the **biadjacency** matrix of the BiCM null model.
#. Calculate the **p-values** of the observed node similarities in the same bipartite layer.

Subsequently, a multiple hypothesis testing of the p-values can be performed. The statistically validated node similarities give rise to an unbiased monopartite projection of the original bipartite network, as illustrated in [Saracco2017]_.

For more detailed explanations of the methods, please refer to [Saracco2017]_, the :ref:`tutorial` and the :ref:`api`.

Obtaining the biadjacency matrix of the BiCM null model
--------------------------------------------------------------------------------

Be ``mat`` a two-dimensional binary NumPy array, which describes the `biadjacency matrix <https://en.wikipedia.org/w/index.php?title=Adjacency_matrix&oldid=751840428#Adjacency_matrix_of_a_bipartite_graph>`_ of an undirected bipartite network. The nodes of the two bipartite layers are ordered along the columns and rows, respectively. In the algorithm, the two layers are identified by the boolean values ``True`` for the **row-nodes** and ``False`` for the **column-nodes**.

Import the module and initialize the Bipartite Configuration Model::

    >>> from src.bicm import BiCM
    >>> cm = BiCM(bin_mat=mat)

To create the biadjacency matrix of the BiCM, use::

    >>> cm.make_bicm()

.. note::

    Note that ``make_bicm`` outputs a *status message* in the console, which informs the user whether the underlying numerical solver has converged to a solution. The function is based on the ``scipy.optimize.root`` routine of the `SciPy package <http://scipy.org>`_ to solve a log-likelihood maximization problem and uses thus the same arguments (except for *fun* and *args*, which are specified in our problem). This means that the user has full control over the selection of a solver, the initial conditions, tolerance, etc.

    As a matter of fact, it may happen that the default function call ``make_bicm()`` results in an unsuccessful solver, which requires adjusting the function arguments. In this case, please refer to the more exhaustive note in the :ref:`tutorial`, the description of the function ``make_bicm`` in the :ref:`api`, and the `scipy.optimize.root documentation <https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.root.html>`_.

The biadjacency matrix of the BiCM null model can be saved in *<filename>*::

    >>> cm.save_biadjacency(filename=<filename>, delim='\t')

By default, the file is saved in a human-readable ``.csv`` format. The matrix can also be saved as a binary NumPy file ``.npy`` by using::

    >>> cm.save_biadjacency(filename=<filename>, binary=True)

If the file is not binary, it should end with, e.g., ``.csv``. If it is binary instead, NumPy automatically appends the ending ``.npy``.

Calculating the p-values of the node similarities
--------------------------------------------------------------------------------

In order to analyze the similarities of the **row-nodes** and to save the p-values of the observed numbers of shared neighbors (i.e. of the :math:`\Lambda`-motifs [Saracco2017]_) in *<filename>*, use::

    >>> cm.lambda_motifs(True, filename=<filename>)

By default, the file is saved as binary NumPy file to reduce disk space, and the format suffix ``.npy`` is appended. If the file should be saved in a human-readable ``.csv`` format, use::

    >>> cm.lambda_motifs(True, filename=<filename>, delim='\t', binary=False)

Analogously for the **column-nodes**, use::

    >>> cm.lambda_motifs(False, filename=<filename>)

or::

    >>> cm.lambda_motifs(False, filename=<filename>, delim='\t', binary=False)

.. note::

    The p-values are saved as a one-dimensional array with index :math:`k \in \left[0, \ldots, \binom{N}{2} - 1\right]` for a bipartite layer of :math:`N` nodes. Please check the section :ref:`output-format` for details regarding the indexing.

Subsequently, the p-values can be used to perform a multiple hypotheses testing of the node similarities and to obtain statistically validated monopartite projections [Saracco2017]_. The p-values are calculated in parallel by default, see :ref:`parallel` for details.

