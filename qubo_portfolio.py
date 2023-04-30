#!/usr/bin/env python
# coding: utf-8

# In[107]:


import yfinance as yf
import pandas as pd
import numpy as np

# Define stock symbols and start and end dates
#symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NFLX', ]
with open('tickers.txt', 'r') as f:
    symbols = [line.strip() for line in f.readlines()]
    symbols = symbols[:8]
start_date = "2022-01-01"
end_date = "2023-01-01"
v = len(symbols)
e = int(v*(v-1)/2)
# Download data from Yahoo Finance
dfs = {}
for i in symbols:
    dfs[i] = yf.download(i, start=start_date, end=end_date)['Close']


# In[108]:


# Merge all dataframes into a single dataframe
df = pd.concat(dfs, axis=1)

# Calculate daily returns
daily_returns = df.pct_change().dropna()


# Calculate expected returns
expected_returns = daily_returns.mean().to_numpy()
Mu = np.diag(expected_returns)

cova = np.cov(daily_returns.T)
std_devs = np.sqrt(np.diag(cova))
corr_matrix = cova / np.outer(std_devs, std_devs)


# Calculate correlation matrix
Sigma = np.corrcoef(daily_returns.T)


# In[112]:


import fireopal as fo
from qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
import math
import networkx as nx
from scipy.optimize import minimize
from operator import itemgetter
import matplotlib.pyplot as plt
from qiskit_ibm_provider import IBMProvider
get_ipython().run_line_magic('matplotlib', 'inline')
import qiskit.tools.jupyter
get_ipython().run_line_magic('qiskit_job_watcher', '')

P = 3

G = nx.gnm_random_graph(v,e)


for i,stock in enumerate(symbols):
    G.add_node(i, label=stock)

n = G.number_of_nodes()
N = int(n)
            
B = 3

B_mtrx = np.zeros((n,n))
np.fill_diagonal(B_mtrx,2*B)
pen_mtrx = 1*(np.ones((n,n))-B_mtrx)
adj_m = np.zeros((n,n))

w = np.array(nx.adjacency_matrix(G, nodelist=range(n))).tolist()
for i,j in G.edges():
    adj_m[i,j]=w[i,j]
    adj_m[j,i]=w[j,i]

vc = [0]*n
for i in range(n):
    for j in range(n):
       # if i<j: (including both upper and lower left triangle)
            if Sigma[i,j] != 0:
                vc[i] += 1            
            
Q = 0.5*cova - Mu + pen_mtrx

nx.draw(G)
#print(adj_m)


# In[89]:


def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1,q2)
    qc.rz(2*gamma, q2)
    qc.cx(q1,q2)
    return qc

def append_z_term(qc, q1, gamma):
    qc.rz(2*gamma, q1)
    return qc

def append_x_term(qc, q1, beta):
    qc.rx(2*beta, q1)
    return qc

def get_cost_operator_circuit(G, gamma):
#    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for i in range(n):
        append_z_term(qc, i, gamma)
    for i in range(n):
        for j in range(n):
            if i<j:
                append_zz_term(qc,i,j,gamma)
#    for (i,j) in G.edges():
#                append_zz_term(qc,i,j,gamma)
#    for i in G.nodes():
#        append_z_term(qc, i, gamma)
    return qc


def get_mixer_operator_circuit(G, beta):
#    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for i in range(N):
        append_x_term(qc, i, beta)
    return qc

def get_qaoa_circuit(G, beta, gamma):
    assert(len(beta) == len(gamma))
    p = len(beta)
#    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)

    # First apply Hadamard layer
    qc.h(range(N))
    
    #Second apply p alternating operators
    for i in range(p):
        qc = qc.compose(get_cost_operator_circuit(G, gamma[i]))  
        qc = qc.compose(get_mixer_operator_circuit(G, beta[i]))
        
    #Finally, measure the result
    qc.barrier(range(N))
    qc.measure(range(N), range(N))

    return qc

def invert_counts(counts):
    return {k[::-1]:v for k,v in counts.items()}

def qubo_obj(x,G):
    summ = 0
    for i in range(n):
        for j in range(n):
            summ += Q[i,j]*int(x[i])*int(x[j])/2
    return summ

def compute_qubo_energy(counts, G):
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = qubo_obj(meas, G)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy/total_counts

def get_black_box_objective(G,p):
    backend = Aer.get_backend('qasm_simulator')
#    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
#    backend = provider.get_backend('ibmq_qasm_simulator')
    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        qc = get_qaoa_circuit(G, beta, gamma)
        counts = execute(qc, backend, seed_simulator = 0).result().get_counts()
        return compute_qubo_energy(invert_counts(counts), G)
    return f


# In[110]:


import time

start = time.time()

p = 4
obj = get_black_box_objective(G, p)

init_point = np.ones(2*p)
res_sample = minimize(obj, init_point, method = 'COBYLA', options = {'maxiter':1000, 'disp':True})

optimal_theta = res_sample['x']
#backend = Aer.get_backend('qasm_simulator')
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_qasm_simulator')
qc = get_qaoa_circuit(G, optimal_theta[:p], optimal_theta[p:])
results = execute(qc, backend).result()
counts = invert_counts(results.get_counts())

end = time.time()
print((end-start),"s")
plot_histogram(counts)


# In[111]:


best_cut, best_solution = min([(qubo_obj(x, G), x) for x in counts.keys()], key=itemgetter(0))
print(f"Best string:{best_solution} with cut:{best_cut}")
colors = ['g' if best_solution[node]=='1' else 'r' for node in G]
#nx.draw(G, node_color = colors)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color=colors)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels={node: G.nodes[node]['label'] for node in G})


# In[623]:





# In[624]:





# In[625]:





# In[626]:





# In[433]:





# In[ ]:





# In[346]:


qc.depth()


# In[9]:


IBMQ.load_account()


# In[329]:


provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
available_backends = [(backend.name(), backend.configuration().n_qubits) for backend in provider.backends()]
print(available_backends)


# In[ ]:




