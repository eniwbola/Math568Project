from math import sin,cos
import numpy as np
import statistics 
import math
import networkx as nx
import pandas as pd
import numpy as np
from RandomGraph import RandomG
import matplotlib.pyplot as plt
from random import random
from Cortical_neuron import Cortical_neuron
from math import exp
import pickle
import statistics
from scipy.stats.stats import pearsonr
import scipy.io
#-------------------------------------------MeanPhaseCoherence---------------------------------------------#

def mpc_network(N,spiketimes_mpc,spikeneurons_mpc):
    # compute mean phase coherence for n cell network
    # compute the every neuron's MPC for possible future use    

        
    # ignore spikes in initial transient for time < 500 
    # spiketimes = spiketimes(spiketimes(:,1)>500,:)
    Cell_Pairs_1=[] # these just take the place of the two columns in the matlab code for the cell pairs
    Cell_Pairs_2=[] #
    
    for i in range(N):
        temp1=[i]*(N-1)  #np.ones(N-1)*(i*1.0)
        temp2=list(range(N))
        temp2.remove(i)
        temp11=[float(v) for v in temp1]
        temp22=[float(u) for u in temp2]
        Cell_Pairs_1= Cell_Pairs_1+temp11 #%map(float,temp1)
        Cell_Pairs_2= Cell_Pairs_2+temp22 #%map(float,temp2)
    
    mpc_cellpairs = [0]*len(Cell_Pairs_1) #np.zeros(N,1)
    	
    def selective_spike_times(spiketimes_mpc,spikeneurons_mpc,Cell_Pairs,iter):
        spikelist=[]
        spikeloc=[]
        for i in range(len(spiketimes_mpc)):        
            if spikeneurons_mpc[i]==Cell_Pairs[iter]:
                spikelist.append(spiketimes_mpc[i])
        return spikelist
    def find_first(cb_spiketimes,ca_spiketimes):
        first_ca_spike=ca_spiketimes[0]
        first_cb_spike_ind=0
        for i in range(len(cb_spiketimes)):
            if cb_spiketimes[i]>first_ca_spike:
                first_cb_spike_ind=i
                return first_cb_spike_ind
                break     
        

    def find_last(cb_spiketimes,ca_spiketimes):
        last_ca_spike=ca_spiketimes[-1]
        last_cb_spike=cb_spiketimes[-1]
        last_cb_spike_ind=0
        if last_ca_spike>last_cb_spike:
            last_cb_spike_ind=len(cb_spiketimes)-1
            #print('heyyy')
            #print(last_ca_spike, last_cb_spike)
            #print('hooo')
            return last_cb_spike_ind
        for i in range(len(cb_spiketimes)):
            if cb_spiketimes[i]>last_ca_spike:
                last_cb_spike_ind=i-1
                return last_cb_spike_ind
                break  
		
   
    for j in range(len(Cell_Pairs_1)):

        # define cell a and cell b
        ca_spiketimes=selective_spike_times(spiketimes_mpc,spikeneurons_mpc,Cell_Pairs_1,j )  #ca_spiketimes = spiketimes(spiketimes(:,2)==cell_pairs(j,1),1)

        
        cb_spiketimes=selective_spike_times(spiketimes_mpc,spikeneurons_mpc,Cell_Pairs_2,j ) #spiketimes(spiketimes(:,2)==cell_pairs(j,2),1)
        
        if ca_spiketimes and cb_spiketimes:  #if ~isempty(ca_spiketimes) and ~isempty(cb_spiketimes)
        
            # determine index of first cell b spike within a cell a cycle
            first_cb_spike = find_first(cb_spiketimes,ca_spiketimes) #find(cb_spiketimes>ca_spiketimes(1),1,'first')
            
            # determine index of last cell b spike within a cell a cycle
            last_cb_spike = find_last(cb_spiketimes,ca_spiketimes) #find(cb_spiketimes<ca_spiketimes(end),1,'last')
            #print("last_cb_spike",last_cb_spike)
            #print("first_cb_spike",first_cb_spike)
            if((last_cb_spike is None) or (first_cb_spike is None) ):
                num_spikes=0
            else:
                num_spikes = last_cb_spike-first_cb_spike+1
            phase = [0]*num_spikes #np.zeros(num_spikes,1) #zeros(num_spikes,1)
            cos_phase =[0]*num_spikes  #np.zeros(num_spikes,1) #zeros(num_spikes,1)
            sin_phase =[0]*num_spikes #np.zeros(num_spikes,1) #zeros(num_spikes,1)

        # compute mean phase of cell b spikes relative to cell a cycle
            k=0
            #print(last_cb_spike)
            if((last_cb_spike is None) or (first_cb_spike is None) or (last_cb_spike == 0)):
                cos_phase=[0,0]
                sin_phase=[0,0]



            else:
                for i in range(first_cb_spike,last_cb_spike+1): #for i=first_cb_spike:last_cb_spike
                    # for each cell b spike determine cell a cycle containing the spike
                    
                    ind_ca_spike2 = find_first(ca_spiketimes,[cb_spiketimes[i]]) #find(ca_spiketimes>=cb_spiketimes(i),1,'first')
                    if ind_ca_spike2 is None:
                        continue
                    #print('ind_ca_spike2',ind_ca_spike2)
                    ca_spike2 = ca_spiketimes[ind_ca_spike2]
                    ca_spike1 = ca_spiketimes[ind_ca_spike2-1]

                    phase[k] = 2*math.pi*(cb_spiketimes[i]-ca_spike1)/(ca_spike2-ca_spike1)
                    cos_phase[k] = cos(phase[k])
                    sin_phase[k] = sin(phase[k])
                    k=k+1        
            if(len(cos_phase)<1):
                mean_cos=0
            else:	
                mean_cos = statistics.mean(cos_phase)

            if(len(sin_phase)<1):
                mean_sin=0
            else:	
                mean_sin = statistics.mean(sin_phase)
            mpc_cellpairs[j] = math.sqrt(mean_cos**2 + mean_sin**2)    

        # remove zero and NaN entries
        #mpc_cellpairs = mpc_cellpairs(mpc_cellpairs~=0)
    #print(mpc_cellpairs)
    mpc_cellpairs=list(filter(lambda mpc_cellpairs: mpc_cellpairs!=0.0, mpc_cellpairs))
    mpc_cellpairs=list(filter(lambda mpc_cellpairs: mpc_cellpairs!=0, mpc_cellpairs))
    mpc_cellpairs=list(filter((0.0).__ne__, mpc_cellpairs))
    #print(np.trim_zeros(mpc_cellpairs))
        #mpc_cellpairs(isnan(mpc_cellpairs))=[]
        #print(mpc_cellpairs)
   
    filter(lambda v: v==v, mpc_cellpairs)  #mpc_cellpairs = mpc_cellpairs[~np.isnan(mpc_cellpairs)]
    mean_mpc = statistics.mean(mpc_cellpairs)
    #mean_mpc=[float(g) for g in mean_mpc]
    float_mpc_cellpairs=[]
    for item in mpc_cellpairs:
        float_mpc_cellpairs.append(float("{0:.2f}".format(item)) )
    #print(float_mpc_cellpairs)
    float_mean_mpc=statistics.mean(float_mpc_cellpairs)	
    
    return [float_mean_mpc,float_mpc_cellpairs]





#--------------------------------------------SpikeGauss---------------------------------------------#



def spikegauss(timestamps,srate,min_timevec,max_timevec,sigma,peak):
#function [spkvec,timevec,updatedpeak]=spikegauss(timestamps,srate,min_timevec,max_timevec,sigma,peak)
    def frange(start1, stop1, step1):
        i=start1-step1
        linlist=[]
        while i<stop1-step1:
            i += step1
            linlist.append(i)
        return linlist

    def histc(X,bins):
        map_to_bins = np.digitize(X,bins)
        r = np.zeros(bins.shape)
        for i in map_to_bins:
            r[i-1] += 1
        return [r, map_to_bins]

    timevec=frange(min_timevec,max_timevec,1.0/srate) # timevec = min_timevec : 1/srate : max_timevec;
    timearray=np.asarray(timevec)
    [spike_count,index] = histc( timestamps, timearray );

    #%gaussian kernel (mean=0; sigma from input)
    gk_mu    = 0;      
    gk_sigma = sigma;
    gk_x = frange(-10.0*sigma+1.0/srate, 10.0*sigma+1.0/srate,1.0/srate )#-10*sigma+1/srate : 1/srate : +10*sigma;
    def gkf(gk_sigma,gk_mu,gk_x):
        return[ 1/(math.sqrt(2*math.pi)* gk_sigma ) * np.exp( - (i-gk_mu)**2 / (2*gk_sigma**2)) for i in gk_x ]
            
    #gk = 1/(sqrt(2*pi)* gk_sigma ) * exp( - (gk_x-gk_mu).^2 / (2*gk_sigma^2));
    gk=gkf(gk_sigma,gk_mu,gk_x)
    #print('yo')
    #print(gk_x)
    if peak==0:
        b=[ i/sum(gk) for i in gk ] #gk=gk/sum(gk);
        gk=b
    else:
        b=[peak*i/max(gk) for i in gk]  #gk=peak*gk/max(gk);
        gk=b
	
    updatedpeak=max(gk)
    npad=len(gk)-1
    full=np.convolve(spike_count,gk,'full')
    first=npad-npad//2
    spkvec=full[first:first+len(spike_count)]
    return [spkvec, timevec ,updatedpeak]

#------------------------------------------SpikeTraces---------------------------------------------#

def spiketraces(N,spiketimes,spikeneurons,min_timevec,max_timevec):

    def selective_spike_times(spiketimes_mpc,spikeneurons_mpc,iter):
        spikelist=[]
        spikeloc=[]
        for i in range(len(spiketimes_mpc)):        
            if spikeneurons_mpc[i]==iter:
                spikelist.append(spiketimes_mpc[i])
        return spikelist

    srate=10      # number of time points per msec 
    #min_timevec=200    # msec
    #max_timevec=5000   # msec
    sigma=2            # msec standard deviation of gaussian
    peak=1             #value of the peak of the gaussian (use peak=0 for gaussian) 
    traces=[]
    for i in range(N):
        stimes=selective_spike_times(spiketimes,spikeneurons,i)
        [trace,timevec,updatedpeak]=spikegauss(stimes,srate,min_timevec,max_timevec,sigma,peak);
        traces.append(trace)
       
    traces_all=np.mean(traces,0)
   

 
 

    return [timevec,traces,traces_all]


#------------------------------------------GolombBurstingMeasure---------------------------------------------#


def GolombBurstingMeasure(n,spiketimes,spikeneurons,min_timevec,max_timevec):
    [timevec,traces,traces_all]=spiketraces(n,spiketimes,spikeneurons,min_timevec,max_timevec)
    sigma=np.zeros((n,1))
    sigma_all=0
    for i in range(n):
        sigma[i]=np.var(traces[i])
        #if i==1:
         #   print(traces[i][600])
    sigma_all=np.var(traces_all)
    #print('traces',traces)
    #print('traces_all',traces_all)
    B=sigma_all/(np.mean(sigma))
    return B
#------------------------------------------CrossCorrelation---------------------------------------------#

def crcorr_network(N,vtraces):
    vtraces=vtraces[0]
    print(vtraces[1])
        
    Cell_Pairs_1=[] # these just take the place of the two columns in the matlab code for the cell pairs
    Cell_Pairs_2=[] #
    for i in range(N):
        temp1=[i]*(N-1)  #np.ones(N-1)*(i*1.0)
        temp2=list(range(N))
        temp2.remove(i)
        temp11=[float(v) for v in temp1]
        temp22=[float(u) for u in temp2]
        Cell_Pairs_1= Cell_Pairs_1+temp11 #%map(float,temp1)
        Cell_Pairs_2= Cell_Pairs_2+temp22 #%map(float,temp2)
    crcorr_cellpairs=[0]*len(Cell_Pairs_1) #np.zeros(len(Cell_Pairs_1),1)
        
    def selective_voltage_traces(vtraces,Cell_Pairs,iter):
        spikelist=[]
        select_voltages=[]
        for i in range(len(vtraces)):        
            if i==Cell_Pairs[iter]:
                return vtraces[i]#select_voltages.append(vtraces[i])
        #selective_voltages_flat=[val for sublist in  select_voltages  for val in sublist ]
        #select_volatges
        #return select_voltages[0]
    print(selective_voltage_traces(vtraces,Cell_Pairs_1,2))
    print(len(selective_voltage_traces(vtraces,Cell_Pairs_1,2)))
    print(len(selective_voltage_traces(vtraces,Cell_Pairs_1,1)))
    
    for j in range(N):#(len(Cell_Pairs_1)-1):
        #crcorr_cellpairs[j]=pearsonr( selective_voltage_traces(vtraces,Cell_Pairs_1,j),selective_voltage_traces(vtraces,Cell_Pairs_2,j))       
        crcorr_cellpairs[j]=np.corrcoef( selective_voltage_traces(vtraces,Cell_Pairs_1,j),selective_voltage_traces(vtraces,Cell_Pairs_2,j))
    print(np.corrcoef( selective_voltage_traces(vtraces,Cell_Pairs_1,2+N),selective_voltage_traces(vtraces,Cell_Pairs_2,3)))
    #for i in range(100):    
    #    print(crcorr_cellpairs[i])
    #print(crcorr_cellpairs)
    #crcorr_cellpairs_flat  = [val for sublist in crcorr_cellpairs for val in sublist]
    #crcorr_cellpairs_real=crcorr_cellpairs[~np.isnan(crcorr_cellpairs)]
    #print(crcorr_cellpairs_flat)
    #for i in range(100):
    #    print(crcorr_cellpairs_flat[i])
    
    crcorr_cellpairs_real=[]
    #for i in crcorr_cellpairs:
    #    print(i)
    #    crcorr_cellpairs_real.append(i[~pd.isnull(i)])
    mean_crcorr=statistics.mean(crcorr_cellpairs)#_real)
    
    return [mean_crcorr, crcorr_cellpairs]





