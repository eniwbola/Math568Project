import networkx as nx
import numpy as np
from RandomGraph import RandomG
from Small_World import smallworld
import Barabasi_Albert 
import matplotlib.pyplot as plt
#from random import random
import random as rp
from random import random
from Cortical_neuron import Cortical_neuron
from math import exp
import pickle
import statistics
import SynMeasures
from datetime import datetime
import scipy

# -------visualization of the network
#network = Barabasi_Albert.ScaleFree(40,1,10,2)
#print(network)
#eigcent=nx.in_eigenvector_centrality(network)
#degcent=nx.in_degree_centrality(network)
#print(['%s %0.2f'%(node,degcent[node]) for node in degcent])
rp.seed(datetime.now())
#network=Barabasi_Albert.ScaleFree(300,.5,5,2)#nx.barabasi_albert_graph(30,2,10)
#networkstr='barabasi_albert_graph(300,2,5)'
#network=RandomG(300,4)#nx.barabasi_albert_graph(30,2,10)
#networkstr='RandomG(300,4)'
network=smallworld(300,.5,3)#nx.barabasi_albert_graph(30,2,10)
networkstr='smallworld(300,.,3)'

#Type=1
#I_amp=-0.03#0.1#0.05
#I_rand=0.0#0.5#0.043
Type=2
I_amp=1.7#1.4
I_rand=0.0#.9#0.5
w = 0.05#0.01 #connection strength

print(Barabasi_Albert.ScaleFree(30,1,10,2))
#print(network.edges)
G = nx.DiGraph(network)
c=nx.number_of_edges(G)/(len(network) *1.0)
#print()

eigcentpre=nx.eigenvector_centrality(G)
eigcent=[]
for i in eigcentpre:
    eigcent.append(eigcentpre[i])

degcentpre=nx.in_degree_centrality(G)
degcent=[]
for i in degcentpre:
    degcent.append(degcentpre[i])

Clustering_Coefspre=nx.clustering(G)
Clustering_Coefs=[]
for i in Clustering_Coefspre:
    Clustering_Coefs.append(Clustering_Coefspre[i])


#for i in eigcent:
#    eigcent.replace(" ",",")
#    eigcent.replace("'","")
   

print(['%s %0.2f'%(node,eigcentpre[node]) for node in eigcentpre])
#print(eigcent)

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
tend=1500.0 #tend = 1000.0
N = len(network) 

delta_t = 0.05 #delta_t = 0.05 ms
I_drive = I_amp*np.ones(N)+I_rand*np.random.rand(N)
E = 0 # 0 mV for excitatory synapses
tau = 0.5 # ms


tpoints = np.arange(t0,tend,delta_t)
vpoints = {}
spike = np.zeros(N,dtype = 'i4')
tspike = np.zeros((N,int(len(tpoints)/10)))




#setup the initial v,z,n,h 
x = np.zeros((N,4))
for i in range(N):
    vpoints[i] = []
    x[i]=[random(),random(),random(),-70+5*random()]
    #x[i]=[random(),random(),random(),-70]
spiketimes_mpc=[];
spikeneurons_mpc=[];

#---------------Simulation-----------------------
for t in tpoints:
    #-----determine the I_syn for ith Neuron
    I_syn = np.zeros(N)
    for i in range(N):
        # Go thru the pre-neuron 
        for j in network[i]:
            if spike[j]: I_syn[i] += w*exp(-(t-tspike[j,spike[j]])/tau)*(E-x[i,3])#(x[i,3]-E)
            
        if x[i,3]<-20: cross = 1
        else: cross = 0
        I = I_syn[i]+I_drive[i]
        k1 = delta_t * Cortical_neuron(x[i],I,Type)
        k2 = delta_t * Cortical_neuron(x[i] + 0.5 * k1,I,Type)
        k3 = delta_t * Cortical_neuron(x[i] + 0.5 * k2,I,Type)
        k4 = delta_t * Cortical_neuron(x[i] + k3,I,Type)
        x[i] += (k1 + 2*k2 + 2*k3 +k4)/6    
        vpoints[i].append(x[i,3])
        if (x[i,3]>-20) and cross: 
            spike[i] += 1
            tspike[i,spike[i]] = t
            
            #tspike_mpc.append([i,t])
            spiketimes_mpc.append(t)
            spikeneurons_mpc.append(i)




#-----------------------Barabasi albert-------------------------#
#------------------------BA/DegreeCentrality vs firing rate-------------------------#














print('eigcent',eigcent)
print('spike',spike)
highest_eigcent=0
highest_eigcentpos=0
for i in range(len(eigcent)):
    if eigcent[i]>highest_eigcent:
        highest_eigcent=eigcent[i]
        highest_eigcentpos=i
print(highest_eigcentpos)     


plt.plot(tpoints,vpoints[0])
plt.xlabel("t")
plt.show()

plt.plot(Clustering_Coefs,spike,'k.')
plt.xlabel("clustering coefficients of neuron"	)
plt.ylabel("number of spikes")
plt.title(networkstr)
plt.show()


plt.plot(spiketimes_mpc,spikeneurons_mpc,'k.')
plt.xlabel("t")
plt.ylabel('neuron I.D.')
plt.title("Raster Plot for"+ " " + networkstr)
plt.show()



plt.plot(eigcent,spike,'k.')
plt.title(networkstr)
plt.xlabel("eigen vector centrality")
plt.ylabel("number of spikes")
plt.show()



plt.plot(degcent,spike,'k.')
plt.title(networkstr)
plt.xlabel("degree centrality")
plt.ylabel("number of spikes")
plt.show()

#scipy.io.savemat('C:/Users/eniwbola/Documents/GitHub/Math568Project/spiketimes_mpc.mat', mdict={'spiketimes_mpc': spiketimes_mpc})
#scipy.io.savemat('C:/Users/eniwbola/Documents/GitHub/Math568Project/spikeneurons_mpc.mat', mdict={'spikeneurons_mpc': spikeneurons_mpc})

[float_mean_mpc,float_mpc_cellpairs]= SynMeasures.mpc_network(N,spiketimes_mpc,spikeneurons_mpc)
print('mpc=',float_mean_mpc)

print('burstingmeasure=',SynMeasures.GolombBurstingMeasure(N,spiketimes_mpc,spikeneurons_mpc))














#f=open('params.txt','wb')
#f.write("type="+ " "  + str(Type) +"\n")
#f.write("I_amp="+ " "  + str(I_amp) +"\n")
#f.write("network=" +" " +networkstr +"\n" )
#f.close()

total_data_file_str= 'total_data_'+networkstr+'_c_c'+str(round(c,2))+'_w_'+str(w)+'I_amp'+str(I_amp)+'Type'+str(Type)


#print(network)
#print(tspike_mpc)
with open('spikedata.pkl','wb') as f:
    pickle.dump([spiketimes_mpc, spikeneurons_mpc],f)

with open('vtraces.pkl','wb') as g:
    pickle.dump([vpoints],g)

with open('total_data.pkl','wb') as e:
    pickle.dump([vpoints,spiketimes_mpc, spikeneurons_mpc],e)



