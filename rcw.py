# coding: utf-8
# Code to perform simple 2D lattice model for RNA collapse
# In[1]:


import numpy as np
import pandas as pd
np.set_printoptions(2)
from scipy.special import comb
import networkx as nx
from random import sample

def sterling_approx(num):
    ''' Used for log(num!)'''
    return num*np.log(num) - num

def calc_n_inner_sites(collapse_degree,rna_length):
    ''' Calculate number of sites within 1 lattice spacing of RNA 
        Inputs
           collapse_degree -- 0/1 stretched/collapsed
            rna_length -- # of RNA sites
        Outputs
            Number of available sites 1 lattice space away 
    '''
    assert collapse_degree <= 1, print('unfeasible collapse degree')

    if collapse_degree == 0:
        return int(2*rna_length+2)
    elif collapse_degree == 1:
        return int(2*(rna_length/2)+4)
    else:
        ds_section = int(collapse_degree*rna_length)
        if ds_section % 2 != 0:
            ds_section += 1
        ss_section = rna_length-ds_section
        return int(2*ss_section+1) + int(2*(ds_section/2)+2)

def rna_graph_generator(collapse_degree,rna_length):
    ''' Generate networkx graph of where rna sites are and connections between
        Inputs
            collapse_degree -- 0/1 stretched/collapsed
            rna_length -- # of RNA sites
        Outputs
            network graph
    '''

    ds_section = int(collapse_degree*rna_length)
    if ds_section % 2 != 0:
        ds_section += 1
    ss_section = rna_length-ds_section

    # Set position of x,y,z coordinates (for 2D z remains fixed at 0)
    rna_sites = []
    x = 0
    y = 0
    z = 0
    for site in range(rna_length):
        if site < ds_section:
            rna_sites.append(np.array([x,y,z]))
            x += 1
            if (site-1) % 2 == 0 and site > 0:
                y += 1
                x = 0
        else:
            rna_sites.append(np.array([x,y,z]))
            y += 1

    # Create graph where each node is an rna_bead
    G = nx.Graph()
    for idx,site1 in enumerate(rna_sites):
        G.add_node(idx)
        G.nodes[idx]['pos'] = site1
        for jdx,site2 in enumerate(rna_sites):
            if jdx <= idx: continue
            if np.linalg.norm(site1-site2) == 1:
                G.add_edge(idx,jdx)
    # To visualize
#     nx.draw_networkx(G)
#     print(G.edges)
    return G

def rna_partition_function(G,Err0,n_ions,kappa):
    ''' Estimate RNA partition function from graph of rna sites based on connections
        Inputs
            G -- nx.Graph() generated from rna_graph_generator()

    '''
    n_neigh = len(G.edges)
    rna_length = len(G.nodes)
    Err = Err0 * np.exp(-kappa*n_ions/rna_length)
    Q = np.exp(-beta*n_neigh*Err)
    F = -kbT*np.log(Q)
    U = n_neigh*Err
    return F,U

def water_partition_function(A,Nw,Nw_inner,Erw):
    '''
        Water Parititon Function for both inner and outer regions
        Inputs:
            A -- # of sites in outer region
            Nw -- # of water in outer region
            Nw_inner -- # of water in inner region
            Erw -- RNA-Water binding energy
        Outputs:
            F -- Free Energy component of water. F = -kbT*np.log(Q), where Q is partition function
    '''
    beta = 1/kbT
    F = Erw*Nw_inner
    F += -kbT*sterling_approx(A)\
             +kbT*sterling_approx(A-Nw)\
             +kbT*sterling_approx(Nw)

    return F

def analytical_ion_partition_function(A,n_ions,Eii):
    '''
        Exact representation of ion partition function, but does not treat it as periodic
        Inputs:
            A -- # of inner sites
            n_ions -- # of ions in inner region
            Eii -- Ion-Ion Energy
        Outputs:
            F -- Free Energy from ion-ion repulsion.
    '''
    Q = 0
    count = np.zeros(n_ions)
    for bins in range(n_ions,0,-1):
        m = n_ions - bins
        n_ways_bin = comb(n_ions-1,bins-1)
        n_ways_nopbc = comb(A-n_ions+1,bins)

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


# Global parameters for running
n_total_sites = 5000
kb = 1
T = 1
kbT = 1/kb/T
beta = 1/kbT
collapse_degrees = np.linspace(0,1,11)


class parameter_setting():
    ''' Parameters that can be changed in the model '''
    def __init__(self):
        self.Eii = 0.3
        self.Erw = -0.1
        self.Err = 0.3
        self.gamma = 0.5
        self.charge_frac = 0.65
        self.kappa = 2
        self.length = 50

    def reset_variables(self):
        self.__init__()

    def change_variable(self,value):
        self


params = parameter_setting()

# Set parameters based on class value
Eii = params.Eii
Erw = params.Erw
Eri = 0
Err = params.Err
gamma = params.gamma
charge_frac = params.charge_frac
kappa = params.kappa
rna_length = int(params.length)
Nw = int(gamma*n_total_sites)
F = np.zeros(len(collapse_degrees))

# Main free energy calculation
for idx,collapse_degree in enumerate(collapse_degrees):
    n_inner_sites = calc_n_inner_sites(collapse_degree,rna_length)
    A = n_total_sites-rna_length-n_inner_sites
    n_ions = int(charge_frac*rna_length)
    Nw_inner = n_inner_sites - n_ions

    G_rna = rna_graph_generator(collapse_degree,rna_length)
    F_RNA = rna_partition_function(G_rna,Err,n_ions,kappa)[0]
    F_ion = analytical_ion_partition_function(n_inner_sites,n_ions,Eii)
    F_water = water_partition_function(A,Nw-Nw_inner,Nw_inner,Erw)

    F[idx] = F_water+F_ion+F_RNA

    # Normalize such that extended section is at F = 0
    if idx == 0:
        F0 = F[0]
    F[idx] = F[idx] - F0

# Most stable state
theta_min = collapse_degrees[np.where(F.min()==F)]
free_ener_min = F.min()



#%% 
# Visualization of results

import matplotlib.pyplot as plt
%matplotlib inline

fig,ax = plt.subplots()
ax.plot(collapse_degrees,F)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$F$ (a.u.)')
ax.set_title('Free Energy vs. Collapse Degree')
fig.set_tight_layout('tight')
# fig.savefig('free_energy.png')
#%%
