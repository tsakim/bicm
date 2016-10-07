# Bipartite Configuration Model

## About
The module contains a Python implementation of the Bipartite Configuration
Model (BiCM), which can be used as a statistical null model for undirected and
binary bipartite networks (see reference \[1, 2\]).

Given the biadjacency matrix of a bipartite network, the corresponding ensemble
average graph is calcutated and expressed in the form of its biadjacency
matrix, where each matrix entry corresponds to the link probability between the
respective nodes.

Furthermore, it is possible to perform a statistical validation of the node
similarities in terms of $\Lambda$-motifs, as described by Saracco et al.
\[1\].
 
## Author 
Mika Straka

## Dependencies
* [ctypes](https://docs.python.org/2/library/ctypes.html)
* [multiprocessing](https://docs.python.org/2/library/multiprocessing.html)
* [scipy](https://www.scipy.org/)
* [numpy](www.numpy.org)
* [poibin](https://github.com/tsakim/poibin) Module for the Poisson Binomial probability distribution 

## Usage
Be `td` a binary matrix in the form of an numpy array. The nodes of the two
distinct bipartite layers are ordered along the columns and rows. In the
following, rows and columns are referred to by the booleans "True" and
"False" and the nodes are called "row-nodes" and "column-nodes",
respectively.

Initialize the Bipartite Configuration Model for the matrix `td` with
```
$ cm = BiCM(bin_mat=td)
```
To create the BiCM by solving the corresponding equation system, use
```
$ cm.make_bicm()
```
To get the Lambda motifs and save the corresponding p-values for the
row-layer nodes in the folder "output", use
```
$ cm.lambda_motifs(True, filename='p_values_True.csv', delim='\t')
```
To get the Lambda motifs and save the corresponding p-values for the
column-layer nodes in the folder "output", use
```
$ cm.lambda_motifs(False, filename='p_values_False.csv', delim='\t')
```
Note that the saving of the files requires the name of the main folder
which contains the folder "src", which itself contains the file bicm.py.
If the folder name is NOT the default "bicm", the function
`self.get_main_dir()` has to be called as
```
self.get_main_dir(main_dir_name=<main folder name>)
```
in` __init__(bin_mat)`.

## Parallel computation

## Testing
- using doctring tests, executed as
```
python -m doctest bicm_tests.txt
```
in the folder `src`.

## References
* \[1\] [Saracco, Di Clemente, Gabrielli, Squartini - Randomizing bipartite networks:
the case of the World Trade Web](http://www.nature.com/articles/srep10595)
* \[2\][Squartini, Garlaschelli - Analytical maximum-likelihood method to detect
patterns in real networks](http://iopscience.iop.org/article/10.1088/1367-2630/13/8/083001)

---
Copyright (C) 2015-2016 Mika Straka 
