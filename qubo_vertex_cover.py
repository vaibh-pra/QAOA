#!/usr/bin/env python
# coding: utf-8

# In[5]:


from qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from operator import itemgetter


# In[7]:


G = nx.gnm_random_graph(5,6)
n = G.number_of_nodes()
Q = np.zeros((n,n))

P = 10
w = np.array(nx.adjacency_matrix(G, nodelist=range(n))).tolist()

for i,j in G.edges():
    Q[i,j]=P/2*w[i,j]
    Q[j,i]=P/2*w[j,i]
for i in range(n):
    for j in range(n):
        if (i,j) not in G.edges() and (j,i) not in G.edges():
            Q[i,j]=0
            Q[j,i]=0

#np.fill_diagonal(Q,-1)
vc = [0]*n
for i in range(n):
    for j in range(n):
       # if i<j: (including both upper and lower left triangle)
            if Q[i,j] != 0:
                vc[i] += 1

nx.draw(G)
for i in range(n):
    for j in range(n):
        if i == j:
            Q[i,j] = 1-(vc[i]*P)
print(vc)
print(Q,G.edges())


# In[8]:


def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1,q2)
    qc.rz(2*gamma, q2)
    qc.cx(q1,q2)
    return qc


# In[134]:


def append_z_term(qc, q1, gamma):
    qc.rz(2*gamma, q1)
    return qc


# In[135]:


def append_x_term(qc, q1, beta):
    qc.rx(2*beta, q1)
    return qc


# In[282]:


def get_cost_operator_circuit(G, gamma):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
#    for i in G.nodes():
#        append_z_term(qc, i, gamma)
#   for i in range(n):
#        for j in range(n):
#            if i<j:
#                append_zz_term(qc,i,j,gamma)
    for i,j in G.edges():
        append_zz_term(qc,i,j,gamma)
#    for i in G.nodes():
#        append_z_term(qc, i, gamma)
    return qc


# In[283]:


def get_mixer_operator_circuit(G, beta):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for n in G.nodes():
        append_x_term(qc, n, beta)
    return qc


# In[284]:


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


# In[285]:


def invert_counts(counts):
    return {k[::-1]:v for k,v in counts.items()}


# In[286]:


def qubo_obj(x,G):
    summ = 0
    for i in range(n):
        for j in range(n):
            summ += Q[i,j]*int(x[i])*int(x[j])
    return summ


# In[287]:


def compute_qubo_energy(counts, G):
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = qubo_obj(meas, G)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy/total_counts


# In[288]:


def get_black_box_objective(G,p):
    backend = Aer.get_backend('qasm_simulator')
    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        qc = get_qaoa_circuit(G, beta, gamma)
        counts = execute(qc, backend, seed_simulator = 10).result().get_counts()
        return compute_qubo_energy(invert_counts(counts), G)
    return f


# In[289]:


import time
start = time.time()

p = 20
obj = get_black_box_objective(G, p)

init_point = np.random.rand(2*p)
res_sample = minimize(obj, init_point, method = 'COBYLA', options = {'maxiter':1000, 'disp':True})

optimal_theta = res_sample['x']
backend = Aer.get_backend('qasm_simulator')
qc = get_qaoa_circuit(G, optimal_theta[:p], optimal_theta[p:])
counts = invert_counts(execute(qc, backend).result().get_counts())


end = time.time()
print((end-start),"s")


# In[290]:


plot_histogram(counts)


# In[291]:


best_cut, best_solution = min([(qubo_obj(x, G), x) for x in counts.keys()], key=itemgetter(0))
print(f"Best string:{best_solution} with cut:{best_cut}")
colors = ['r' if best_solution[node]=='1' else 'b' for node in G]
nx.draw(G, node_color = colors)
for x in counts.keys():
    print(x, qubo_obj(x,G))


# In[292]:


qc = get_qaoa_circuit(G, [np.pi/4], [np.pi/3])
qc.draw()


# In[293]:


qc.depth()


# In[ ]:




