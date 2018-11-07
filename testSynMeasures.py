#!/usr/bin/python3.5

import networkx as nx
import numpy as np
from RandomGraph import RandomG
import matplotlib.pyplot as plt
from random import random
from Cortical_neuron import Cortical_neuron
from math import exp
import pickle
import statistics as stat
import SynMeasures
import scipy.io



n=300

with open('spikedata.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    spiketimes_mpc, spikeneurons_mpc = pickle.load(f)

with open('vtraces.pkl','rb') as g:
    vpoints=pickle.load(g)

#[whatever_data]=pickle.load( open( "spikedata.pkl", "rb" ) )
scipy.io.savemat('/home/bolaji/hw/Math568/NetworkProject/Math568Project/spiketimes_mpc.mat', mdict={'spiketimes_mpc': spiketimes_mpc})
scipy.io.savemat('/home/bolaji/hw/Math568/NetworkProject/Math568Project/spikeneurons_mpc.mat', mdict={'spikeneurons_mpc': spikeneurons_mpc})


#print(spiketimes_mpc)

#SynMeasures.mpc_network(300,spiketimes_mpc,spikeneurons_mpc )
#print(SynMeasures.mpc_network(300,spiketimes_mpc,spikeneurons_mpc ))





#[mean_mpc,mpc_cellpairs]=SynMeasures.mpc_network(n,spiketimes_mpc,spikeneurons_mpc )# so this seems to work tested with against matlab

#[timevec,traces,traces_all]=SynMeasures.spiketraces(n,spiketimes_mpc,spikeneurons_mpc)
#print('traces',traces[1][600])
#print('traces_all',traces_all[7344])
#print('length traces all',len(traces_all))

#B=SynMeasures.GolombBurstingMeasure(n,spiketimes_mpc,spikeneurons_mpc)
#print('B',B)

[mean_crcorr, crcorr_cellpairs]=SynMeasures.crcorr_network(n,vpoints)
#print('mean_crcorr', mean_crcorr)
#print('crcorr_cellpairs',crcorr_cellpairs)



#print(SynMeasures.mpc_network(n,spiketimes_mpc,spikeneurons_mpc ))





#-------------------individua--test---------------------------------------#
#Im checking this against spiketraces.m at /home/bolaji/hw/Math568/NetworkProject/MatlabCode/spiketrace
stimes=[261.65,352.1, 442.75,553,624.05,714.7,805.35,896.0, 986.7]
srate=10
min_timevec=200
max_timevec=2000
sigma=2
peak=1

#[spkvec, timevec ,updatedpeak]=SynMeasures.spikegauss(stimes,srate,min_timevec,max_timevec,sigma,peak)

#print("spkvec[600]",spkvec[600],np.mean(spkvec)) # so I'm pretty sure tmy spikegauss is rght
#print('timevec',timevec)






