Bipartite Configuration Model for Python - Documentation
================================================================================

The Bipartite Configuration Model (BiCM) is a statistical null model for binary
bipartite networks [Squartini2011]_, [Saracco2015]_. It offers an unbiased method for analyzing node
similarities and obtaining statistically validated monopartite projections
[Saracco2017]_.

The BiCM belongs to a series of entropy-based null models for binary bipartite 
networks, see also

* `BiPCM <https://github.com/tsakim/bipcm>`_ - Bipartite Partial Configuration Model
* `BiRG <https://github.com/tsakim/birg>`_ - Bipartite Random Graph

Please consult the original articles for details about the underlying methods
and applications to user-movie and international trade databases
[Saracco2017]_, [Straka2017]_.

An example case is illustrated in the :ref:`tutorial`.

How to cite
--------------------------------------------------------------------------------

If you use the ``bicm`` module, please cite its `location on Github
<https://github.com/tsakim/bicm>`_ and the original articles [Saracco2015]_ and
[Saracco2017]_.

References
````````````````````````````````````````````````````````````````````````````````

.. [Saracco2015] `F. Saracco, R. Di Clemente, A. Gabrielli, T. Squartini, Randomizing bipartite networks: the case of the World Trade Web, Scientific Reports 5, 10595 (2015) <http://www.nature.com/articles/srep10595>`_

.. [Saracco2017] `F. Saracco, M. J. Straka, R. Di Clemente, A. Gabrielli, G. Caldarelli, and T. Squartini, Inferring monopartite projections of bipartite networks: an entropy-based approach, New J. Phys. 19, 053022 (2017) <http://stacks.iop.org/1367-2630/19/i=5/a=053022>`_

.. [Squartini2011] `T. Squartini, D. Garlaschelli, Analytical maximum-likelihood method to detect patterns in real networks, New Journal of Physics 13, (2011) <http://iopscience.iop.org/article/10.1088/1367-2630/13/8/083001>`_

.. [Straka2017] `M. J. Straka, G. Caldarelli, F. Saracco, Grand canonical validation of the bipartite international trade network, Phys. Rev. E 96, 022306 (2017) <https://doi.org/10.1103/PhysRevE.96.022306>`_


Getting Started
================================================================================

.. toctree::
   :maxdepth: 2

   ./source/overview
   ./source/quickstart
   ./source/tutorial
   ./source/testing
   ./source/parallel
   ./source/src
   ./source/license
   ./source/contact

Indices and tables
================================================================================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

