#!/usr/bin/env python
# coding: utf-8

# In[31]:


from qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from operator import itemgetter
import matplotlib.pyplot as plt


# In[49]:


G = nx.Graph()
#E = [(0, 1, 3), (0, 2, 2), (0, 3, 10), (1, 2, 6), (1, 3, 2), (2, 3, 6)]

G = nx.gnm_random_graph(6,15)

#G.add_weighted_edges_from(E)

n = G.number_of_nodes()
N = int(n-1)**2

Q = np.ones((n,n))

for i,j in G.edges():
    G.edges[i, j]['weight'] = round(np.random.rand()*100)

#P = 8
w = np.array(nx.adjacency_matrix(G, nodelist=range(n))).tolist()
#w = nx.adjacency_matrix(G, nodelist=range(n))

for i,j in G.edges():
        Q[i,j]=w[i,j]
        Q[j,i]=w[j,i]
#print(str(Q[i][j]))

for i in range(n):
    for j in range(n):
        if (i,j) not in G.edges() and (j,i) not in G.edges():
            Q[i,j]=100
            Q[j,i]=100

vc = [0]*n
for i in range(n):
    for j in range(n):
       # if i<j: (including both upper and lower left triangle)
            if Q[i,j] != 0:
                vc[i] += 1            
            
#np.fill_diagonal(Q,-1)

for i in range(n):
    for j in range(n):
        if i == j:
            Q[i,j] = 0

nx.draw(G)
print(Q,G.edges(),vc)


# In[33]:


def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1,q2)
    qc.rz(2*gamma, q2)
    qc.cx(q1,q2)
    return qc


# In[34]:


def append_z_term(qc, q1, gamma):
    qc.rz(2*gamma, q1)
    return qc


# In[35]:


def append_x_term(qc, q1, beta):
    qc.rx(2*beta, q1)
    return qc


# In[36]:


def get_cost_operator_circuit(G, gamma):
#    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for i in range(N):
        append_z_term(qc, i, gamma)
    for i in range(N):
        for j in range(N):
            if i<j:
                append_zz_term(qc,i,j,gamma)
    return qc


# In[37]:


def get_mixer_operator_circuit(G, beta):
#    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for n in range(N):
        append_x_term(qc, n, beta)
    return qc


# In[38]:


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


# In[39]:


def invert_counts(counts):
    return {k[::-1]:v for k,v in counts.items()}


# In[40]:


def qubo_obj(string,G):
    
    A = 500
    n = G.number_of_nodes()
    g = n-1
    # Convert the string to a 1D array of integers.
    array1d = np.array([int(i) for i in string])
    
    # Reshape the 1D array into a 2D array of size 2x2
    x = array1d.reshape((g,g))
    
    
    summ1, row_sum, col_sum, summ5 = 0,0,0,0
    summ2, summ4, summ3, tot_sum = 0,0,0,0 
    
    #Qubo term
    for i in range(g):
        for j in range(g):
            for k in range(g):
                for l in range(g):
                    if i!=j and k != l:
                        if (i,j) in G.edges():
                            summ1 += x[i][k]*x[j][l]*Q[i][j]/2
                        if (i,j) not in G.edges():
                            summ4 += x[i][k]*x[j][l]/2
    
                        
    #Row and Col summations
    row_sum = np.sum(x, axis = 1)
    col_sum = np.sum(x, axis = 0)
    
    #Penalty terms
    summ2 = np.sum((1-row_sum)**2)
    summ3 = np.sum((1-col_sum)**2)
    
    # Penalty for Hamiltonian cycle
    for i in range(g):
        for j in range(g):
            if i != j:
                # Penalize if vertex i is visited twice or not visited at all
                summ5 += (x[i][j] * x[i][j] - x[i][j])
    
    tot_sum = summ1 + A*(summ2 + summ3 + summ4 + summ5)
    
    return tot_sum


# In[41]:


def compute_qubo_energy(counts, G):
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = qubo_obj(meas, G)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy/total_counts


# In[42]:


def get_black_box_objective(G,p):
    backend = Aer.get_backend('qasm_simulator')
    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        qc = get_qaoa_circuit(G, beta, gamma)
        counts = execute(qc, backend, seed_simulator = 10).result().get_counts()
        return compute_qubo_energy(invert_counts(counts), G)
    return f


# In[ ]:


import time
start = time.time()

p = 1
obj = get_black_box_objective(G, p)

init_point = np.ones(2*p)
res_sample = minimize(obj, init_point, method = 'SLSQP', options = {'maxiter':1000, 'disp':True})

optimal_theta = res_sample['x']
backend = Aer.get_backend('qasm_simulator')
qc = get_qaoa_circuit(G, optimal_theta[:p], optimal_theta[p:])
counts = invert_counts(execute(qc, backend).result().get_counts())


end = time.time()
print((end-start),"s")


# In[1302]:


#plot_histogram(counts)


# In[48]:


pos = nx.spring_layout(G)
options = {
    "with_labels": True,
    "font_weight": "bold",
    "font_color": "white",
    "node_size": 2000,
    "width": 2
}

min_engy, optimal_string = min([(qubo_obj(x, G), x) for x in counts.keys()])
#print(min_engy, optimal_string)
op_str = optimal_string[::-1]

array1d = np.array([int(i) for i in optimal_string])
mtrx1 = array1d.reshape((n-1,n-1))
#print(mtrx1, optimal_string)

str_by_vertex = [optimal_string[i:i + n - 1] for i in range(0, len(optimal_string) + 1, n-1)]
salesman_walk = '0'.join(str_by_vertex) + '0' * (n-1) + '1'
array1d = np.array([int(i) for i in salesman_walk])
mtrx = array1d.reshape((n,n))
print(mtrx, salesman_walk)

all_zero_cols = np.where(np.all(mtrx == 0, axis=0))[0]
all_zero_rows = np.where(np.all(mtrx == 0, axis=1))[0]

mtrx[all_zero_rows, all_zero_cols] = 1

mtrx_str = ''.join(map(str, mtrx.flatten()))

str_by_sw = [mtrx_str[i:i + n-1] for i in range(0, len(mtrx_str) + 1, n - 1)]
solution = {i:t for i in range(n) for t in range(n) if mtrx_str[i * n + t] == '1'}

print(mtrx, mtrx_str)

distance = sum([G[u][v]["weight"] if solution[u] == (solution[v] + 1) % n 
                   or solution[v] == (solution[u] + 1) % n else 0
                 for (u, v) in G.edges])
print("The walk found by parameterized quantum circuit:", solution, "with distance", distance)

label_dict = {i: str(i) + ", " + str(t) for i, t in solution.items()}
edge_color = ["red" if solution[u] == (solution[v] + 1) % n
              or solution[v] == (solution[u] + 1) % n else "black"
              for (u, v) in G.edges]

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
for i, a in enumerate(ax):
    a.axis('off')
    a.margins(0.20)
nx.draw(G, pos=pos, labels=label_dict, edge_color=edge_color, ax=ax[0], **options)
nx.drawing.nx_pylab.draw_networkx_edge_labels(G, pos=pos, ax=ax[0], edge_labels=nx.get_edge_attributes(G, 'weight'))
nx.drawing.nx_pylab.draw_networkx_edge_labels(G, pos=pos, ax=ax[1], edge_labels=nx.get_edge_attributes(G, 'weight'))
plt.axis("off")
plt.show()


# In[ ]:


qc = get_qaoa_circuit(G, [np.pi/4], [np.pi/3])
qc.draw()
qc.depth()


# In[1065]:


#for x in counts.keys():
#    print(x, qubo_obj(x,G))


# In[ ]:




