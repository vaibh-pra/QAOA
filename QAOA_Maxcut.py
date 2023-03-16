#!/usr/bin/env python
# coding: utf-8

# In[156]:


from qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
import networkx as nx
from scipy.optimize import minimize


# In[18]:


G = nx.Graph()
G.add_edges_from([[0,3],[0,4],[1,3],[1,4],[2,3],[2,4]])
nx.draw(G, pos=nx.bipartite_layout(G,[0,1,2]))


# In[21]:


def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1,q2)
    qc.rz(2*gamma, q2)
    qc.cx(q1,q2)
    return qc


# In[23]:


def append_x_term(qc, q1, beta):
    qc.rx(2*beta, q1)
    return qc


# In[74]:


def get_cost_operator_circuit(G, gamma):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for i,j in G.edges():
        append_zz_term(qc,i,j,gamma)
    return qc


# In[122]:


def get_mixer_operator_circuit(G, beta):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for n in G.nodes():
        append_x_term(qc, n, beta)
    return qc


# In[131]:


def get_qaoa_circuit(G, beta, gamma):
    assert(len(beta) == len(gamma))
    p = len(beta)
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    # First apply Hadmard layer
    qc.h(range(N))
    #Second apply p alternating operators
    for i in range(p):
        qc = qc.compose(get_cost_operator_circuit(G, gamma[i]))
        qc = qc.compose(get_mixer_operator_circuit(G, beta[i]))
    #Finally, measure the result
    qc.barrier(range(N))
    qc.measure(range(N), range(N))
#    print(len(beta))
    return qc


# In[126]:


def invert_counts(counts):
    return {k[::-1]:v for k,v in counts.items()}


# In[92]:


def maxcut_obj(x, G):
    cut = 0
    for i,j in G.edges():
        if x[i] != x[j]:
            cut -= 1
    return cut


# In[128]:


def compute_maxcut_energy(counts, G):
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = maxcut_obj(meas, G)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy/total_counts


# In[181]:


def get_black_box_objective(G,p):
    backend = Aer.get_backend('qasm_simulator')
    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        qc = get_qaoa_circuit(G, beta, gamma)
        counts = execute(qc, backend, seed_simulator = 5).result().get_counts()
        return compute_maxcut_energy(invert_counts(counts), G)
    return f


# In[197]:


p = 1
obj = get_black_box_objective(G, p)
init_point = np.ones(2*p)
res_sample = minimize(obj, init_point, method = 'COBYLA', options = {'maxiter':1000, 'disp':True})
#res_sample


# In[198]:


optimal_theta = res_sample['x']
backend = Aer.get_backend('qasm_simulator')
qc = get_qaoa_circuit(G, optimal_theta[:p], optimal_theta[p:])
counts = invert_counts(execute(qc, backend).result().get_counts())
plot_histogram(counts)
#optimal_theta[p:]


# In[194]:


np.ones(p)


# In[ ]:




