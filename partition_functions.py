# coding: utf-8
# Expression for partition functions of simple 2D lattice model
# In[1]: 
import numpy as np
import pandas as pd
from scipy.special import comb
import networkx as nx
from random import sample

def sterling_approx(num):
    ''' Used for log(num!)'''
    return num*np.log(num) - num

def rna_partition_function(G,Err0,n_ions,kappa,kbT):
    ''' Estimate RNA partition function from graph of rna sites based on connections
        Inputs
            G -- nx.Graph() generated from rna_graph_generator()

    '''
    beta = 1/kbT
    n_neigh = len(G.edges)
    rna_length = len(G.nodes)
    Err = Err0 * np.exp(-kappa*n_ions/rna_length)
    Q = np.exp(-beta*n_neigh*Err)
    F = -kbT*np.log(Q)
    U = n_neigh*Err
    return F,U

def water_partition_function(A,Nw_outer,Nw_inner,Erw,kbT):
    '''
        Water Parititon Function for both inner and outer regions
        Inputs
            A -- # of outer sites
            Nw_outer -- # of water in bulk region
            Nw_inner -- # of water in inner region
            Erw -- RNA-water energy interaction
    '''
    F = Erw*Nw_inner
    F += -kbT*sterling_approx(A)\
             +kbT*sterling_approx(A-Nw_outer)\
             +kbT*sterling_approx(Nw_outer)

    return F

def ion_rna_partition_function(n_ions,Eri,kbT):
    ''' Compute interaction between ions and rna '''
    beta = 1/kbT
    Q = np.exp(-beta*Eri*n_ions)
    F = -kbT*np.log(Q)
    return F

def analytical_ion_partition_function(A,n_ions,Eii,kbT):
    '''
        Exact representation of ion partition function, but does not treat it as periodic
        Inputs
            A -- # of inner sites
            n_ions -- # of ions in inner region
    '''
    beta = 1/kbT
    Q = 0
    count = np.zeros(n_ions)
    for bins in range(n_ions,0,-1):
        m = n_ions - bins
        n_ways_bin = comb(n_ions-1,bins-1)
        n_ways_nopbc = comb(A-n_ions+1,bins)
        # print('Bins: {}; m {}; W1 {}; W2 {}'.format(bins,m,n_ways_bin,n_ways_nopbc))

        count[m] += n_ways_bin*n_ways_nopbc

        # Account for the case where two ions are at the edge (since recursive don't need to loop)
        if m != n_ions-1:
            n_fix = 2
            n_ways_pbc = comb(A-n_ions-n_fix+1,bins-n_fix)
            count[m] -= n_ways_bin*n_ways_pbc
            count[m+1] += n_ways_bin*n_ways_pbc

    for idx,bins in enumerate(range(n_ions,0,-1)):
        m = n_ions - bins
        Q += count[idx]*np.exp(-beta*m*Eii)
    F = -kbT*np.log(Q)
    return F

