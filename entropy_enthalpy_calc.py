# coding: utf-8
# In[1]:

import numpy as np
import pandas as pd
np.set_printoptions(2)
from scipy.special import comb
import networkx as nx
from random import sample
import partition_functions

def calc_n_inner_sites(collapse_degree,rna_length):
    ''' Calculate number of sites within 1 lattice spacing of RNA '''
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
    '''

    ds_section = int(collapse_degree*rna_length)
    if ds_section % 2 != 0:
        ds_section += 1
    ss_section = rna_length-ds_section

    # Set position of x,y,z coordinates
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


# Check to see if lower entropic costs
rna_length = 50
n_total_sites = 5000
kb = 1
T = 1
kbT = 1/kb/T
beta = 1/kbT
collapse_degrees = np.linspace(0,1,11)


class parameter_setting():
    ''' Parameters that can be changed in the model '''
    def __init__(self):
        self.Eii = 0.1
        self.Erw = -0.05
        self.Err = 0.1
        self.Eri = -0.1
        self.gamma = 0.55
        self.charge_frac = 0.6
        self.kappa = 1
        self.length = 50

    def reset_variables(self):
        self.__init__()

    def change_variable(self,value):
        self

count = 0

temperatures = [0.90,0.95,1.00,1.05,1.10]
charge_fracs = np.linspace(0.5,1,20)
free_energy = np.zeros((len(temperatures),len(charge_fracs),len(collapse_degrees)))
df = {}
params = parameter_setting()
param_list = vars(params).items()

for ix1,temperature in enumerate(temperatures):
    kbT = kb*temperature
    beta = 1/temperature
    df[temperature] = {}
    for ix2,charge_frac in enumerate(charge_fracs):
        df[temperature][charge_frac] = {}
        df[temperature][charge_frac]['water'] = np.zeros((len(collapse_degrees)))
        df[temperature][charge_frac]['ion'] = np.zeros((len(collapse_degrees)))
        df[temperature][charge_frac]['rna'] = np.zeros((len(collapse_degrees)))
        # df[temperature][charge_frac]['ion_rna'] = np.zeros((len(collapse_degrees)))
        df[temperature][charge_frac]['total'] = np.zeros((len(collapse_degrees)))
        params = parameter_setting()
        params.reset_variables()
        Eii = params.Eii
        Erw = params.Erw
        Eri = params.Eri
        Err = params.Err
        gamma = params.gamma
        kappa = params.kappa
        rna_length = int(params.length)

        Nw = int(gamma*n_total_sites)
        for ix3,collapse_degree in enumerate(collapse_degrees):
            n_inner_sites = calc_n_inner_sites(collapse_degree,rna_length)
            A = n_total_sites-rna_length-n_inner_sites
            n_ions = int(charge_frac*rna_length)
            Nw_inner = n_inner_sites - n_ions
            Nw_outer = Nw-Nw_inner

            G_rna = rna_graph_generator(collapse_degree,rna_length)
            F_RNA = partition_functions.rna_partition_function(G_rna,Err,n_ions,kappa,kbT)[0]

            F_ion = partition_functions.analytical_ion_partition_function(n_inner_sites,n_ions,Eii,kbT)
            F_ion_rna = partition_functions.ion_rna_partition_function(n_ions,Eri,kbT)
            F_water = partition_functions.water_partition_function(A,Nw_outer,Nw_inner,Erw,kbT)

            free_energy[ix1,ix2,ix3] = F_water+F_ion+F_RNA+F_ion_rna
            if ix3 == 0:
                F0 = free_energy[ix1,ix2,ix3]
            free_energy[ix1,ix2,ix3] = free_energy[ix1,ix2,ix3] - F0
            df[temperature][charge_frac]['water'][ix3] = F_water
            df[temperature][charge_frac]['ion'][ix3] = F_ion
            df[temperature][charge_frac]['rna'][ix3] = F_RNA
            # df[temperature][charge_frac]['ion_rna'][ix3] = F_ion_rna
            df[temperature][charge_frac]['total'][ix3] = F_water+F_ion+F_RNA+F_ion_rna
        # theta_min = collapse_degrees[np.where(F.min()==F)]
        # free_ener_min = F.min() - F[0]



# In[2]
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from matplotlib import cm
plt.rcParams['font.size'] = 14

fig,ax = plt.subplots()
# sns.heatmap(df)

plt1 = ax.contourf(collapse_degrees,charge_fracs*rna_length,free_energy[0],levels=15)
fig.colorbar(plt1)
ax.set_xlabel(r'$\theta$',fontsize=16)
ax.set_ylabel(r'$N_i$',fontsize=16)
fig.set_tight_layout('tight')
fig.savefig('num_ions_theta_landscape.png',dpi=300)
#%%
fig,ax = plt.subplots()
# sns.heatmap(df)

ix_charge_fracs = 4
ax.plot(collapse_degrees,free_energy[2,ix_charge_frac,:])
ax.set_xlabel(r'$\theta$',fontsize=16)
ax.set_ylabel(r'$F(\theta)$',fontsize=16)
fig.set_tight_layout('tight')
fig.savefig('free_energy.png',dpi=300)
#%%
S = {}
U = {}
temp2 = temperatures[3]
temp0 = temperatures[1]
beta2 = 1/temp2
beta0 = 1/temp0
keys = list(df[temperature][charge_frac].keys())

for ix2,charge_frac in enumerate(charge_fracs):
    U[charge_frac] = {}
    S[charge_frac] = {}
    for key in keys:
        U[charge_frac][key] = (df[temp2][charge_frac][key]*beta2-df[temp0][charge_frac][key]*beta0)/(beta2-beta0)
        S[charge_frac][key] = -(df[temp2][charge_frac][key]-df[temp0][charge_frac][key])/(temp2-temp0)
            
# S = {}
# U = {}

fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(10,8))
ax = np.ravel(ax)
charge_frac1 = charge_fracs[ix_charge_frac]
total = np.zeros((3,len(collapse_degrees)))
for ik,key in enumerate(keys):
    y = np.copy(-S[charge_frac1][key])
    y -= y[0]
    ax[0].plot(y,label=key)
    total[0] -= y
    y = np.copy(U[charge_frac1][key])
    y -= y[0]
    ax[1].plot(y,label=key)
    total[1] += y
    # y = np.copy(U[charge_frac1][key]-S[charge_frac1][key])
    # y -= y[0]
    # ax[2].plot(y,label=key)
    # total[2] += y

ax[0].legend(fontsize=12)
ax[0].set_ylabel(r'-TS $(k_bT)$')
ax[1].set_ylabel(r'U $(k_bT)$')

U1 = U[charge_frac1]['total']; U1 -= U1[0]
S1 = S[charge_frac1]['total']; S1 -= S1[0]
ax[2].plot(U1,label='U',color='green')
ax[2].plot(-S1,label='-TS',color='blue')
ax[2].plot(U1-S1,label='F',color='magenta')
ax[2].set_ylabel(r'F $(k_bT)$')
ax[2].legend(fontsize=12)

fig.delaxes(ax[-1])
fig.set_tight_layout('tight')
for aa in ax:
    aa.set_xlabel(r'$\theta$')
fig.savefig('energy_species.png',dpi=300)

# %%
fig,ax = plt.subplots()
U1 = {}
temp2 = temperatures[3]
temp0 = temperatures[1]
beta2 = 1/temp2
beta0 = 1/temp0
keys = list(df[temperature][charge_frac].keys())

for it,temperature in enumerate(temperatures):
    U1[temperature] = {}
    try: temp2 = temperatures[it+1]
    except: None
    temp0 = temperatures[it-1]
    if it == 0:
        temp0 = temperatures[0]
    elif it == 4:
        temp2 = temperatures[4]


    for ix2,charge_frac in enumerate(charge_fracs):
        U1[temperature][charge_frac] = {}
        for key in keys:
            U1[temperature][charge_frac][key] = (df[temp2][charge_frac][key]*beta2-df[temp0][charge_frac][key]*beta0)/(beta2-beta0)

temp2 = 1.05
temp0 = 0.95
y2 = U1[temp2][charge_frac1]['total']
y0 = U1[temp0][charge_frac1]['total']
Cv = (y2-y0) / (temp2-temp0)
Cv -= Cv[0]
ax.plot(collapse_degrees,Cv)
ax.set_ylabel(r'$C_V$')
ax.set_xlabel(r'$\theta$')
fig.set_tight_layout('tight')
fig.savefig('specific_heat.png',dpi=300)
# %%
