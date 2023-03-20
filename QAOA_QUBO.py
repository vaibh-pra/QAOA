#!/usr/bin/env python
# coding: utf-8

# In[920]:


from qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from operator import itemgetter


# In[921]:


#G = nx.Graph()
G = nx.gnm_random_graph(5,6)
#G.add_edges_from([[0,3],[0,4],[1,3],[1,4],[2,3],[2,4]])
w = np.array(nx.adjacency_matrix(G, nodelist=range(G.number_of_nodes()))).tolist()
n = G.number_of_nodes()
Q = np.ones((n,n))

for i,j in G.edges():
        Q[i,j]=w[i,j]
        Q[j,i]=w[j,i]
for i in range(n):
    for j in range(n):
        if (i,j) not in G.edges() and (j,i) not in G.edges():
            Q[i,j]=-1
            Q[j,i]=-1
            
np.fill_diagonal(Q,1)
nx.draw(G)
print(Q,G.edges())


# In[922]:


def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1,q2)
    qc.rz(2*gamma, q2)
    qc.cx(q1,q2)
    return qc


# In[923]:


def append_z_term(qc, q1, gamma):
    qc.rz(2*gamma, q1)
    return qc


# In[924]:


def append_x_term(qc, q1, beta):
    qc.rx(2*beta, q1)
    return qc


# In[925]:


def get_cost_operator_circuit(G, gamma):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for i in G.nodes():
        append_z_term(qc, i, gamma)
    for i,j in G.edges():
        append_zz_term(qc,i,j,gamma)
    return qc


# In[926]:


def get_mixer_operator_circuit(G, beta):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for n in G.nodes():
        append_x_term(qc, n, beta)
    return qc


# In[927]:


def get_qaoa_circuit(G, beta, gamma):
    assert(len(beta) == len(gamma))
    p = len(beta)
    N = G.number_of_nodes()
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


# In[928]:


def invert_counts(counts):
    return {k[::-1]:v for k,v in counts.items()}


# In[929]:


def qubo_obj(x, G):
    n = G.number_of_nodes()
    c = []
    for i in range(n):
        coef = 0
        for j in range(n):
            coef += Q[i][j]
        c.append(coef)
    summ = 0
    for i,j in G.edges():
        if i<j:
            summ += 2*int(x[i])*int(x[j])-int(x[i])-int(x[j]) #Not incorporating graph weights.
    return summ


# In[930]:


def compute_qubo_energy(counts, G):
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = qubo_obj(meas, G)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy/total_counts


# In[931]:


def get_black_box_objective(G,p):
    backend = Aer.get_backend('qasm_simulator')
    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        qc = get_qaoa_circuit(G, beta, gamma)
        counts = execute(qc, backend, seed_simulator = 5).result().get_counts()
        return compute_qubo_energy(invert_counts(counts), G)
    return f


# In[932]:


p = 2
obj = get_black_box_objective(G, p)
init_point = np.ones(2*p)
res_sample = minimize(obj, init_point, method = 'COBYLA', options = {'maxiter':1000, 'disp':True})


# In[933]:


optimal_theta = res_sample['x']
backend = Aer.get_backend('qasm_simulator')
qc = get_qaoa_circuit(G, optimal_theta[:p], optimal_theta[p:])
counts = invert_counts(execute(qc, backend).result().get_counts())
plot_histogram(counts)


# In[456]:


best_cut, best_solution = min([(qubo_obj(x, G), x) for x in counts.keys()], key=itemgetter(0))
print(f"Best string:{best_solution} with cut:{best_cut}")
colors = ['r' if best_solution[node]=='1' else 'b' for node in G]
nx.draw(G, node_color = colors)


# In[ ]:




