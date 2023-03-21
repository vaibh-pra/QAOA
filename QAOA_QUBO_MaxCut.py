#!/usr/bin/env python
# coding: utf-8

# In[1245]:


from qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from operator import itemgetter


# In[1348]:


#G = nx.Graph()
G = nx.gnm_random_graph(5,6)
#G.add_edges_from([[0,3],[0,4],[1,3],[1,4],[2,3],[2,4]])
n = G.number_of_nodes()
Q = np.ones((n,n)) #Initialize the matrix

w = np.array(nx.adjacency_matrix(G, nodelist=range(n))).tolist()

for i,j in G.edges():
        Q[i,j]=2*w[i,j]
        Q[j,i]=2*w[j,i]
for i in range(n):
    for j in range(n):
        if (i,j) not in G.edges() and (j,i) not in G.edges():
            Q[i,j]=0
            Q[j,i]=0
#np.fill_diagonal(Q,-1)

nx.draw(G)
#print(Q,G.edges())


# In[1349]:


def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1,q2)
    qc.rz(2*gamma, q2)
    qc.cx(q1,q2)
    return qc


# In[1350]:


def append_z_term(qc, q1, gamma):
    qc.rz(2*gamma, q1)
    return qc


# In[1351]:


def append_x_term(qc, q1, beta):
    qc.rx(2*beta, q1)
    return qc


# In[1352]:


def get_cost_operator_circuit(G, gamma):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for i in G.nodes():
        append_z_term(qc, i, gamma)
    for i,j in G.edges():
        append_zz_term(qc,i,j,gamma)
    return qc


# In[1353]:


def get_mixer_operator_circuit(G, beta):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for n in G.nodes():
        append_x_term(qc, n, beta)
    return qc


# In[1354]:


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


# In[1355]:


def invert_counts(counts):
    return {k[::-1]:v for k,v in counts.items()}


# In[1356]:


def qubo_obj(x, G):
    n = G.number_of_nodes()
    c = []
    for i in range(n):
        coef = 0
        for j in range(n):
            coef = Q[i,j]
        c.append(coef)
    summ = 0
    for i,j in G.edges():
        if i<j:
            summ += Q[i,j]*int(x[i])*int(x[j])-int(x[i])**2-int(x[j])**2 # = x^T Q x
    return summ


# In[1357]:


def compute_qubo_energy(counts, G):
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = qubo_obj(meas, G)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy/total_counts


# In[1358]:


def get_black_box_objective(G,p):
    backend = Aer.get_backend('qasm_simulator')
    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        qc = get_qaoa_circuit(G, beta, gamma)
        counts = execute(qc, backend, seed_simulator = 10).result().get_counts()
        return compute_qubo_energy(invert_counts(counts), G)
    return f


# In[1367]:


p = 5
obj = get_black_box_objective(G, p)
init_point = np.ones(2*p)
res_sample = minimize(obj, init_point, method = 'COBYLA', options = {'maxiter':1000, 'disp':True})


# In[1368]:


optimal_theta = res_sample['x']
backend = Aer.get_backend('qasm_simulator')
qc = get_qaoa_circuit(G, optimal_theta[:p], optimal_theta[p:])
counts = invert_counts(execute(qc, backend).result().get_counts())
plot_histogram(counts)


# In[1370]:


best_cut, best_solution = min([(qubo_obj(x, G), x) for x in counts.keys()], key=itemgetter(0))
print(f"Best string:{best_solution} with cut:{best_cut}")
colors = ['r' if best_solution[node]=='1' else 'b' for node in G]
nx.draw(G, node_color = colors)


# In[1371]:


for x in counts.keys():
    print(x, qubo_obj(x,G))


# In[ ]:




