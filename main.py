import networkx as nx
import numpy as np
from RandomGraph import RandomG
import matplotlib.pyplot as plt
from random import random
from Cortical_neuron import Cortical_neuron
from math import exp,ceil
import pickle
import statistics
from Barabasi_Albert import ScaleFree
from Small_World import smallworld
import SynMeasures

n = 300
network1 = ScaleFree(n)
G1 = nx.DiGraph(network1)
c = nx.number_of_edges(G1)/n
# get the mean degree of the scale free network

out_RAD_EE = int(ceil(c/2))
P_EE = c/(2*out_RAD_EE)
network2 = smallworld(n,P_EE,out_RAD_EE)
G2 = nx.DiGraph(network2)

network3 = RandomG(n,c)
G3 = nx.DiGraph(network3)


# -------visualization of the network
'''
options = {
    'node_color': 'Yellow',
    'node_size': 100,
    'edge_color': 'black',    
    'width': 1,    
    'arrows': True,
}

plt.figure(figsize=(16,16))
plt.title('Scale-Free Network')
nx.draw_kamada_kawai(G1, **options)
plt.show()

plt.figure(figsize=(16,16))
plt.title('Small-World Network')
nx.draw_circular(G2, **options)
plt.show()
plt.figure(figsize=(16,16))
plt.title('Random Graph')
nx.draw_kamada_kawai(G3, **options)
plt.show()
'''
#---------------------------------------
#------Neural Network Simulation--------
wpoints = np.linspace(0.01,0.2,3)
mpcpoints = np.zeros((3,len(wpoints)))
burstingpoints = np.zeros((3,len(wpoints)))
for rounds in range(3):
    if rounds == 0:
        network = network1
        nametag = '_scale_free_'
        print(nametag)
        print('Average Clustering Coefficient = ',nx.average_clustering(G1))
        print('Reciprocity = ',nx.reciprocity(G1))
        Closeness_Centrality = nx.closeness_centrality(G1)
        #print(Closeness_Centrality)
        print('Average Closeness Centrality = ',statistics.mean([Closeness_Centrality[i] for i in range(n)]))


    elif rounds == 1:
        network = network2
        nametag = '_small_world_'
        print(nametag)

    else:
        network = network3
        nametag = '_random_graph_'
        print(nametag)
        
#----------Initialization---------------
    t0 = 0.0
    tend=5000.0 #tend = 1000.0
    N = len(network) 
    print('N = ',N)

    delta_t = 0.2 #delta_t = 0.05 ms
    I_drive = 1.3*np.ones(N)+0.02*np.random.rand(N)
    E = 0 # 0 mV for excitatory synapses
    tau = 0.5 # ms
    windex = 0
    for w in wpoints: #connection strength

        tpoints = np.arange(t0,tend,delta_t)
        vpoints = {}
        spike = np.zeros(N,dtype = 'i4')
        tspike = np.zeros((N,int(len(tpoints)/10)))

        #setup the initial v,z,n,h 
        x = np.zeros((N,4))
        for i in range(N):
            vpoints[i] = []
            x[i]=[random(),random(),random(),-70+40*random()]

        spiketimes_mpc=[]
        spikeneurons_mpc=[]

        #---------------Simulation-----------------------
        for t in tpoints:
            #-----determine the I_syn for ith Neuron
            I_syn = np.zeros(N)
            for i in range(N):
                # Go thru the pre-neuron 
                for j in network[i]:
                    if spike[j]: I_syn[i] += w*exp(-(t-tspike[j,spike[j]-1])/tau)*(x[i,3]-E)
                if x[i,3]<-20: cross = 1
                else: cross = 0
                I = -I_syn[i]+I_drive[i]
                k1 = delta_t * Cortical_neuron(x[i],I)
                k2 = delta_t * Cortical_neuron(x[i] + 0.5 * k1,I)
                k3 = delta_t * Cortical_neuron(x[i] + 0.5 * k2,I)
                k4 = delta_t * Cortical_neuron(x[i] + k3,I)
                x[i] += (k1 + 2*k2 + 2*k3 +k4)/6    
                vpoints[i].append(x[i,3])
                if (x[i,3]>-20) and cross:                 
                    tspike[i,spike[i]] = t
                    spike[i] += 1
                    #tspike_mpc.append([i,t])
                    spiketimes_mpc.append(t)
                    spikeneurons_mpc.append(i)
        #print(tspike_mpc)
        str1 = 'spikedata_n_' + str(N)+'_c_'+str(round(c,2))+'_w_'+str(round(w,2))+nametag+'.pkl'
        with open(str1,'wb') as f:
            pickle.dump([spiketimes_mpc, spikeneurons_mpc],f)
        
        #str2 = 'vtraces_n_' + str(N)+'_c_'+str(round(c,2))+'_w_'+str(w)+nametag+'.pkl'
        #with open(str2,'wb') as g:
        #    pickle.dump(vpoints,g)    
        mpcpoints[rounds][windex] = SynMeasures.mpc_network(N,spiketimes_mpc,spikeneurons_mpc)[0]
        burstingpoints[rounds][windex] = SynMeasures.GolombBurstingMeasure(N,spiketimes_mpc,spikeneurons_mpc)
        windex += 1
        #print('MPC = ',)
        #print('BurstingMeasure = ',SynMeasures.GolombBurstingMeasure(N,spiketimes_mpc,spikeneurons_mpc))
        for ii in range(N):
            plt.plot(tspike[ii][:spike[ii]],ii*np.ones(spike[ii]),'b.')
            plt.title((nametag)+str(round(w,2)))
        #plt.show()
        plt.savefig(('rasterplot_' + str(N)+'_c_'+str(round(c,2))+'_w_'+str(round(w,2))+nametag+'.png'))
        plt.close()
plt.plot(wpoints,mpcpoints[0],label = 'Scale Free')
plt.plot(wpoints,mpcpoints[1],label = 'Small World')
plt.plot(wpoints,mpcpoints[2],label = 'Random Graph')
plt.legend()
plt.title('Mean phase Coherence')
plt.xlabel('connection strength')
plt.ylabel('MPC')
plt.show()
plt.plot(wpoints,burstingpoints[0],label = 'Scale Free')
plt.plot(wpoints,burstingpoints[1],label = 'Small World')
plt.plot(wpoints,burstingpoints[2],label = 'Random Graph')
plt.legend()
plt.title('Bursting Measure')
plt.xlabel('connection strength')
plt.ylabel('Bursting Measure')
plt.show()



