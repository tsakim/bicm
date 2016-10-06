# coding: utf-8

# # Bipartite Configuration Model

# Read trade data and create the corresponding bipartite configuration model.
# Script
# Add the main project folder to the PYTHONPATH in order to use the
# modules
import sys, os
import time
import numpy as np
main_module_path = os.path.abspath(os.path.join('..'))
if main_module_path not in sys.path:
    sys.path.append(main_module_path)
import src.itn_birg as birg

def get_main_dir(main_dir_name='ITN'):
    """Return the absolute path to the main directory.

    :param main_dir_name: name of the main directory of the program.
    :type main_dir_name: string
    """
    s = os.getcwd()
    dirpath = s[:s.index(main_dir_name) + len(main_dir_name) + 1]
    return dirpath

for yr in range(2001, 2011, 1):
    print "----------"
    print "+++Load Trade Data..."
    print "...year: ", yr
    print "+++Start Time: ", time.asctime(time.localtime(time.time()))
    file_name = get_main_dir()
    file_name += '/data/hs2007_data_cleaned/hs2007_MM_' + str(yr) + '.dat'
    td = np.loadtxt(file_name)
    rg = birg.BiRG(td, yr)
    print "...get lambda2_motifs and p-values for the countries"
    rg.lambda2_motifs_main(bip_set=True, write=True)
    print "...get lambda2_motifs and p-values for the products"
    rg.lambda2_motifs_main(bip_set=False, write=True)
    print "+++End Time: ", time.asctime(time.localtime(time.time()))
    print "----------"
