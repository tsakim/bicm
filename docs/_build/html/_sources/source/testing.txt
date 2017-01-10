.. _testing:

Testing
--------------------------------------------------------------------------------

The methods in the ``bicm`` module  have been implemented using `doctests
<https://docs.python.org/2/library/doctest.html>`_. To run the tests,
execute::

    >>> python -m doctest bicm_tests.txt

from the folder `src` in the command line. If you want to run the tests in
verbose mode, use::

    >>> python -m doctest -v bicm_tests.txt

Note that `bicm.py` and `bicm_tests.txt` have to be in the same directory to
run the test.

