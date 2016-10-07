# Bipartite Configuration Model

## About
The module contains a Python implementation of the Bipartite Configuration
Model (BiCM), which can be used as a statistical null model for undirected and
binary bipartite networks (see reference \[1, 2\]).

Given the biadjacency matrix of a bipartite network, the corresponding ensemble
average graph is calcutated and expressed in the form of its biadjacency
matrix, where each matrix entry corresponds to the link probability between two
nodes in the two distinct layers.

To address the question of node similarity within one bipartite layer, it is
possible to perform a statistical validation of the number of common nearest
neighbors and to calculate the p-values of the correspective Lambda-motifs, as
described by Saracco et al. \[1\].
 
## Author 
Mika Straka

## Dependencies
* [ctypes](https://docs.python.org/2/library/ctypes.html)
* [multiprocessing](https://docs.python.org/2/library/multiprocessing.html)
* [scipy](https://www.scipy.org/)
* [numpy](www.numpy.org)
* [poibin](https://github.com/tsakim/poibin) Module for the Poisson Binomial probability distribution 

## Usage
Be `td` a two-dimensional binary numpy array describing the biadjacency matrix
of an undirected bipartite network. The nodes of the two distinct bipartite
layers are ordered along the columns and rows, respectively. In the algorithm, 
the two layers are identified by the boolean values `True` for row-nodes and `False` for column-nodes.

Import the module
```python
$ from bicm import BiCM
```
and initialize the Bipartite Configuration Model for the matrix `td` with
```python
$ cm = BiCM(bin_mat=td)
```
To create the BiCM by solving the corresponding equation system \[1\], use
```python
$ cm.make_bicm()
```
The biadjacency matrix of the BiCM null model can be saved in the folder
`bicm/output/` as
```python
$ cm.save_matrix(cm.adj_matrix, <filename>, delim='\t'>
```
where `<filename>' should end with `.csv` or similar and the delimiter `delim`
can be freely chosen, the default value being `\t`.

In order to analyze the Lambda-motifs and save the corresponding p-values for
the row-layer nodes in the folder `bicm/output/`, use
```python
$ cm.lambda_motifs(True, filename='p_values_True.csv', delim='\t')
```
To get the Lambda motifs and save the corresponding p-values for the
column-layer nodes in the folder `bicm/output/`, use 
```python
$ cm.lambda_motifs(False, filename='p_values_False.csv', delim='\t')
```

### NB: Main folder
Note that the saving of the files requires the name of the main directory,
which contains the folder `src` and thus the file `src/bicm.py`.
If the folder name is *not* the default `bicm`, the BiCM instance has to be initialized as
```python
$ cm = BiCM(bin_mat=td, main_dir=<main directory>)
```

## Parallel computation
The module uses the Python multiprocessing package in order to execute the
calculation of the p-values in parallel. The number of parallel processes
depends on the number of CPUs of the work station (see variable "numprocs" in
the method `BiCM.get_pvalues_q`. 
If the calculation should not be performed in parallel, use
```python
$ cm.lambda_motifs(True, parallel=False)
```python
and respectively
```
$ cm.lambda_motifs(False, parallel=False)
```

## Testing
- using doctring tests, executed as
```
python -m doctest bicm_tests.txt
```
in the folder `src`.

## References
* \[1\] [Saracco, Di Clemente, Gabrielli, Squartini - Randomizing bipartite networks:
the case of the World Trade Web](http://www.nature.com/articles/srep10595)
* \[2\] [Squartini, Garlaschelli - Analytical maximum-likelihood method to detect
patterns in real networks](http://iopscience.iop.org/article/10.1088/1367-2630/13/8/083001)

---
Copyright (C) 2015-2016 Mika Straka 
