import networkx as nx
import numpy as np
from RandomGraph import RandomG
import matplotlib.pyplot as plt
from random import random
from Cortical_neuron import Cortical_neuron
from math import exp
import pickle
import statistics

# -------visualization of the network
network = RandomG()
G = nx.DiGraph(network)
print
options = {
    'node_color': 'Yellow',
    'node_size': 100,
    'edge_color': 'black',    
    'width': 1,    
    'arrows': True,
}

plt.figure(figsize=(16,16))
nx.draw_kamada_kawai(G, **options)
plt.show()
#---------------------------------------
#------Neural Network Simulation--------
#----------Initialization---------------
t0 = 0.0
tend=1000.0 #tend = 1000.0
N = len(network) 

delta_t = 0.05 #delta_t = 0.05 ms
I_drive = 1.3*np.ones(N)+0.02*np.random.rand(N)
E = 0 # 0 mV for excitatory synapses
tau = 0.5 # ms
w = 0.01 #connection strength

tpoints = np.arange(t0,tend,delta_t)
vpoints = {}
spike = np.zeros(N,dtype = 'i4')
tspike = np.zeros((N,int(len(tpoints)/10)))

#setup the initial v,z,n,h 
x = np.zeros((N,4))
for i in range(N):
    vpoints[i] = []
    x[i]=[random(),random(),random(),-70+40*random()]

spiketimes_mpc=[];
spikeneurons_mpc=[];

#---------------Simulation-----------------------
for t in tpoints:
    #-----determine the I_syn for ith Neuron
    I_syn = np.zeros(N)
    for i in range(N):
        # Go thru the pre-neuron 
        for j in network[i]:
            if spike[j]: I_syn[i] += w*exp(-(t-tspike[j,spike[j]])/tau)*(x[i,3]-E)
        if x[i,3]<-20: cross = 1
        else: cross = 0
        I = I_syn[i]+I_drive[i]
        k1 = delta_t * Cortical_neuron(x[i],I)
        k2 = delta_t * Cortical_neuron(x[i] + 0.5 * k1,I)
        k3 = delta_t * Cortical_neuron(x[i] + 0.5 * k2,I)
        k4 = delta_t * Cortical_neuron(x[i] + k3,I)
        x[i] += (k1 + 2*k2 + 2*k3 +k4)/6    
        vpoints[i].append(x[i,3])
        if (x[i,3]>-20) and cross: 
            spike[i] += 1
            tspike[i,spike[i]] = t
            #tspike_mpc.append([i,t])
            spiketimes_mpc.append(t)
            spikeneurons_mpc.append(i)
#print(tspike_mpc)
with open('spikedata.pkl','wb') as f:
    pickle.dump([spiketimes_mpc, spikeneurons_mpc],f)

with open('vtraces.pkl','wb') as g:
    pickle.dump([vpoints],g)

plt.plot(tpoints,vpoints[0])
plt.xlabel("t")
plt.show()

