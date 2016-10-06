# Analysis of the International Trade Network

## Description

The python scripts can be used to analyze the international trade network (ITN)
and its projection on the countries and products. An appropriate null model, the
bipartite configuration model, can be created in order to analyze the links
of the projections and the correspective link weight probabilities.
 
## Author 
Mika Straka

## The newest version
The newest version can be obtained from the author.

## Dependencies
Python packages:
- igraph
- scipy
- numpy
- matplotlib
- poibin  - can be obtained from ....

## Folder contents
- analysis/     results of the network analysis
- aux/          auxiliary files, such as output of code profiling for comparison
- data/         input data for the trade network (i.e. BACI data)
- output/       output of the network analysis routines, such as .graphml files

## Tests
- using doctring tests, executed as
python -m doctest bicm_tests.txt
in the folder src.

## References
Squartini, Garlaschelli - Analytical maximum-likelihood method to detect
patterns in real networks,
DOI: 10.1088/1367-2630/13/8/083001

Saracco, Di Clemente, Gabrielli, Squartini - Randomizing bipartite networks:
the case of the World Trade Web,
DOI: 10.1038/srep10595

--------------------------------------------------------------------------------
Copyright (C) 2015-2016 Mika Straka 
