#!/usr/bin/env python
# coding: utf-8

# In[1003]:


from qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
import math
import networkx as nx
from scipy.optimize import minimize
from operator import itemgetter
import matplotlib.pyplot as plt


# In[1316]:


#G = nx.Graph()
#E = [(0, 1, 3), (0, 2, 2), (0, 3, 10), (1, 2, 6), (1, 3, 2), (2, 3, 6)]

G = nx.gnm_random_graph(4,6)
#Tree = tree_decomposition(G)
#G.add_weighted_edges_from(E)

n = G.number_of_nodes()
N = int(n**2)
m = math.ceil(math.sqrt(N))
print(m,N)
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
            Q[i,j]=0
            Q[j,i]=0

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


# In[1005]:


def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1,q2)
    qc.rz(2*gamma, q2)
    qc.cx(q1,q2)
    return qc


# In[1260]:


def append_z_term(qc, q1, gamma):
    qc.rz(2*gamma, q1)
    return qc


# In[1261]:


def append_x_term(qc, q1, beta):
    qc.rx(2*beta, q1)
    return qc


# In[1262]:


def get_cost_operator_circuit(G, gamma):
#    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    
    for i in range(N):
        append_z_term(qc, i, gamma)

    edge_list = list(G.edges())
#    print(edge_list)

    for k in range(0, N, n):
        for i, j in edge_list:
            i += k
            j += k
            if i < j:
#                print(i,j)
                append_zz_term(qc, i, j, gamma)
#    for i in range(N):
#        for j in range(N):
#            if i<j:
#                print(i,j)
#                append_zz_term(qc, i, j, gamma)
    return qc


# In[1212]:


qc = get_qaoa_circuit(G, [np.pi/4], [np.pi/3])


# In[1010]:


def get_mixer_operator_circuit(G, beta):
#    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for i in range(N):
        append_x_term(qc, i, beta)
    return qc


# In[1011]:


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


# In[1012]:


def invert_counts(counts):
    return {k[::-1]:v for k,v in counts.items()}


# In[1013]:


def qubo_obj(string,G):
    
    A = 500
    n = G.number_of_nodes()
    g = n
    
    # Convert the string to a 1D array of integers using list comprehension
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
                    if i<j and k != l:
                        if (i,j) in G.edges():
                            summ1 += x[i][k]*x[j][l]*Q[i][j]
                        if (i,j) not in G.edges():
                            summ4 += x[i][k]*x[j][l]
                        
    #Row and Col summations
    row_sum = np.sum(x, axis = 1)
    col_sum = np.sum(x, axis = 0)
    
    #Penalty terms
    summ2 = np.sum((1-row_sum)**2)
    summ3 = np.sum((1-col_sum)**2)
    
   
    tot_sum = summ1 + A*(summ2 + summ3 + summ4)
    
    return tot_sum


# In[1014]:


print(qubo_obj('0000100010100000100000100', G))
print(range(n))


# In[1015]:


def compute_qubo_energy(counts, G):
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = qubo_obj(meas, G)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy/total_counts


# In[1016]:


def get_black_box_objective(G,p):
    backend = Aer.get_backend('qasm_simulator')
    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        qc = get_qaoa_circuit(G, beta, gamma)
        counts = execute(qc, backend, seed_simulator = 0).result().get_counts()
        return compute_qubo_energy(invert_counts(counts), G)
    return f


# In[1320]:


import time

start = time.time()

p = 4
obj = get_black_box_objective(G, p)

init_point = np.random.rand(2*p)
res_sample = minimize(obj, init_point, method = 'COBYLA', options = {'maxiter':1000, 'disp':True})

optimal_theta = res_sample['x']
backend = Aer.get_backend('qasm_simulator')
qc = get_qaoa_circuit(G, optimal_theta[:p], optimal_theta[p:])
counts = invert_counts(execute(qc, backend).result().get_counts())


end = time.time()
print((end-start),"s")


# In[1144]:


plot_histogram(counts)


# In[1323]:


pos = nx.spring_layout(G)
options = {
    "with_labels": True,
    "font_weight": "bold",
    "font_color": "white",
    "node_size": 2000,
    "width": 2
}

#min_engy, optimal_string = min([(qubo_obj(x, G), x) for x in counts.keys()])

energy_x_list = [(qubo_obj(x, G), x) for x in counts.keys()]
# Find the minimum energy and corresponding value of x in the list
min_engy, _ = min(energy_x_list)

# Find all the strings that give the minimum energy
optimal_strings = [x for (energy, x) in energy_x_list if energy == min_engy]
#engy = [energy for (energy, x) in energy_x_list if energy == min_engy]

# Print the optimal strings
#print("Strings that give minimum energy:", optimal_strings)

#optimal_string = optimal_string[::-1]
def optimal_distance(optimal_string):
    array1d = np.array([int(i) for i in optimal_string])
#mtrx1 = array1d.reshape((n-1,n-1))
#print(mtrx1, optimal_string)

#str_by_vertex = [optimal_string[i:i + n - 1] for i in range(0, len(optimal_string) + 1, n-1)]
#salesman_walk = '0'.join(str_by_vertex) + '0' * (n-1) + '1'
#array1d = np.array([int(i) for i in salesman_walk])
    mtrx = array1d.reshape((n,n))
#print(mtrx, optimal_string)

    all_zero_cols = np.where(np.all(mtrx == 0, axis=0))[0]
    all_zero_rows = np.where(np.all(mtrx == 0, axis=1))[0]

    mtrx[all_zero_rows, all_zero_cols] = 1

    mtrx_str = ''.join(map(str, mtrx.flatten()))

    str_by_sw = [mtrx_str[i:i + n-1] for i in range(0, len(mtrx_str) + 1, n - 1)]
    solution = {i:t for i in range(n) for t in range(n) if mtrx_str[i * n + t] == '1'}

#print(mtrx, mtrx_str)

    distance = sum([G[u][v]["weight"] if solution[u] == (solution[v] + 1) % n 
                   or solution[v] == (solution[u] + 1) % n else 0
                 for (u, v) in G.edges])
    
    return distance

for i in range(len(optimal_strings)):
    optimal_string = optimal_strings[i]
#    print(optimal_string, optimal_distance(optimal_string))

min_distance, a = min([(optimal_distance(y), y) for y in optimal_strings])

solution = {i:t for i in range(n) for t in range(n) if a[i * n + t] == '1'}

    
print("The walk found by parameterized quantum circuit:", solution, "with distance", min_distance)

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


# In[1322]:


qc = get_qaoa_circuit(G, [np.pi/4], [np.pi/3])
qc.draw()
qc.depth()
num_single_qubit_gates = qc.count_ops()['h'] + qc.count_ops()['rx']
num_two_qubit_gates = qc.count_ops()['cx']
num_gates = num_single_qubit_gates + num_two_qubit_gates
print(num_gates)


# In[1065]:


#for x in counts.keys():
#    print(x, qubo_obj(x,G))


# In[362]:


print(qubo_obj('1001100010', G))


# In[700]:


A = 10
n = G.number_of_nodes()
g=n-1
summ1 = 0
string1, string2 ='',''
string3 = ''
for i in range(g+1):
    for j in range(g+1):
        for k in range(g+1):
            for l in range(g+1):
                if i<j and k != l:
                    if (i,j) in G.edges():
                        string1 = 'x_' + str(i) + '_' + str(k)
                        string2 = 'x_' + str(j) + '_' + str(l)
                        q = str(Q[i][j])
                        string3 += q + ' ' + string1 + ' ' + string2 + ' + '
print(string3)
                        #summ1 += x[i][j]*x[k][l]
#print(summ1)
summ2 = 0
summ3 = 0
string5 = ''
string7 = ''
for i in range(n):
    string4 = ''
    string6 = ''
    insumm1 = 0
    insumm2 = 0
    for k in range(n):
        string4 += 'x_'+str(k)+'_'+str(i) + '  '
        string6 += 'x_'+str(i)+'_'+str(k) + '  '

    string5 += str(1)+'-'+string4
    string7 += str(1)+'-'+string6


# In[ ]:


# Penalty for Hamiltonian cycle
   for i in range(g):
       for j in range(g):
           if i != j:
               # Penalize if vertex i is visited twice or not visited at all
               summ5 += (2*x[i][j] - x[i][j])

