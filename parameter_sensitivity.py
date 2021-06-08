# coding: utf-8
# Vary hyperparameters 1 at a time, Sensitivity Analysis
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


param_space = {
    'Eii' : np.linspace(0,1,5),
    'Erw' : np.linspace(0,-0.5,5),
    'Err' : np.linspace(0,1,5),
    'gamma' : np.linspace(0.4,0.9,5),
    'charge_frac' : np.linspace(0.5,1,5),
    'kappa' : np.linspace(0,5,5),
    'length' : np.linspace(31,70,5,dtype=int),
}


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


df_columns = [x for x in param_space]
df_columns.append('theta_min')
df_columns.append('free_energy_min')
for idx,_ in enumerate(collapse_degrees):
    df_columns.append(idx)

df = pd.DataFrame(columns=df_columns)
params = parameter_setting()
param_list = vars(params).items()

for key,value in param_space.items():
    params = parameter_setting()
    params.reset_variables()
    for v in value:
        setattr(params,key,v)
        Eii = params.Eii
        Erw = params.Erw
        Eri = params.Eri
        Err = params.Err
        gamma = params.gamma
        charge_frac = params.charge_frac
        kappa = params.kappa
        rna_length = int(params.length)


        df.loc[count] = 0
        df.loc[count]['Eii','Erw','Err','gamma','charge_frac','kappa','length'] = [Eii,Erw,Err,gamma,charge_frac,kappa,rna_length]
        Nw = int(gamma*n_total_sites)
        F = np.zeros(len(collapse_degrees))
        for idx,collapse_degree in enumerate(collapse_degrees):
            n_inner_sites = calc_n_inner_sites(collapse_degree,rna_length)
            A = n_total_sites-rna_length-n_inner_sites
            n_ions = int(charge_frac*rna_length)
            Nw_inner = n_inner_sites - n_ions
            Nw_outer = Nw - Nw_inner

            G_rna = rna_graph_generator(collapse_degree,rna_length)
            F_RNA = partition_functions.rna_partition_function(G_rna,Err,n_ions,kappa,kbT)[0]
            F_ion = partition_functions.analytical_ion_partition_function(n_inner_sites,n_ions,Eii,kbT)
            F_water = partition_functions.water_partition_function(A,Nw_outer,Nw_inner,Erw,kbT)
            F_ion_rna = partition_functions.ion_rna_partition_function(n_ions,Eri,kbT)

            F[idx] = F_water+F_ion+F_RNA+F_ion_rna
            if idx == 0:
                F0 = F[0]
            df.loc[count][idx] = F[idx] - F[0]
        theta_min = collapse_degrees[np.where(F.min()==F)]
        free_ener_min = F.min() - F[0]

        df.loc[count]['theta_min'] = float(theta_min)
        df.loc[count]['free_energy_min'] = free_ener_min
        # df.loc[count]['free_energy_min'] = F[4]-F[0]
        count += 1

# df.to_pickle('phaseSpace_grid.pkl')
# np.savetxt('phaseSpace.txt',df.values,fmt='%4.3f')



import matplotlib.pyplot as plt
%matplotlib inline

fig,ax = plt.subplots(nrows=3,ncols=3)
ax = np.ravel(ax)
count = 0
plot_idx = 0
y = []
y1 = []
x = []
key_count = 0
titles = [r'$E_{ii}$',r'$E_{rw}$',r'$E_{rr}$',r'$\gamma$',r'$\zeta$',\
          r'$\kappa$',r'$R_l$']

for key,value in param_space.items():
    for jdx,v in enumerate(value):
        x.append(v)
        y.append(df.loc[count].theta_min)
        y1.append(df.loc[count].free_energy_min)
        # print(key,v,df.loc[count].free_energy_min-df.loc[33].free_energy_min)
        if jdx == 1:
            x1 = v
            x2 = getattr(params,key)
            dy = df.loc[count].free_energy_min-df.loc[33].free_energy_min
            print(key,dy/(x1-x2))
        count += 1
        
    ax[plot_idx].plot(x,y1,'o')
    ax[plot_idx].set_title(titles[plot_idx])
    ax[plot_idx].set_ylabel(r'$F_{min}$')
    plot_idx += 1
    x = []
    y = []
    y1 = []

fig.set_tight_layout('tight')
# fig.savefig('parameter_free_energy.png')
#%%
sens = {}
y = []
for key in param_space.keys():
    x1 = param_space[key][1]
    x3 = param_space[key][3]
    f1 = float(df.loc[df[key]==x1]['free_energy_min'])
    f3 = float(df.loc[df[key]==x3]['free_energy_min'])
    xmax = param_space[key][-1]
    xmin = param_space[key][0]
    sens[key] = (f1-f3)/(x1-x3) * 1/np.abs(xmax-xmin)
    y.append(sens[key])

fig,ax = plt.subplots()
keys = list(param_space.keys())
titles = [r'$E_{ii}$',r'$E_{rw}$',r'$E_{rr}$',r'$\gamma$','$\zeta$','$\kappa$',r'$R_l$']
x0 = np.arange(len(keys))
ax.bar(x0,y)
ax.set_xticks(x0)
ax.set_xticklabels(titles)

fig.set_tight_layout('tight')
fig.savefig('parameter_sensitivity.png',dpi=300)
# %%
