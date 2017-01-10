.. _parallel:

Parallel Computation
--------------------------------------------------------------------------------

Since the calculation of the p-values is computationally demanding, the
``bicm`` module uses the Python `multiprocessing
<https://docs.python.org/2/library/multiprocessing.html>`_ package by default
for this purpose.  The number of parallel processes depends on the number of
CPUs of the work station (see variable ``numprocs`` in the method
:func:`BiCM.get_pvalues_q` in the :ref:`api`). 

If the calculation should **not** be performed in parallel, use::

    >>> cm.lambda_motifs(<bool>, parallel=False)

instead of::

    >>> cm.lambda_motifs(<bool>)

