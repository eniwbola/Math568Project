from pylab import plot, xlabel, show, legend
from numpy import array, arange
from math import exp

def Cortical_neuron(x,I,type=2):
#I in uA/cm2

    z = x[0]
    h = x[1]
    n = x[2]
    v = x[3]

    m_infnty = (1 + exp((-v-30.0)/9.5))**-1
    h_infnty = (1 + exp((v+53.0)/7.0))**-1
    n_infnty = (1 + exp((-v-30.0)/10))**-1
    z_infnty = (1 + exp((-v-39.0)/5.0))**-1

    tau_h = 0.37 + 2.78*(1 + exp((v+40.5)/6))**-1
    tau_n = 0.37 + 1.85*(1 + exp((v+27.0)/15))**-1

    fh = (h_infnty - h)/tau_h
    fn = (n_infnty - n)/tau_n
    fz = (z_infnty - z)/75.0

    C = 1 #uf/cm2
    v_Na = 55.0 #mV
    v_K = -90 #mV
    v_L = -60 #mV
    g_Na = 24 #mS/cm2
    g_Kdr = 3.0 #mS/cm2   	 
    g_L = 0.02 #mS/cm2
    if type == 1: #high Ach
        g_Ks = 0
    elif type == 2: #low Ach
        g_Ks = 1.5 #mS/cm2
    
    fv = (1/C)*(g_Na*(m_infnty**3)*h*(v_Na-v)+g_Kdr*(n**4)*(v_K-v)+g_Ks*z*(v_K - v)+g_L*(v_L-v)+I)
    
    f = array([fz, fh, fn, fv],float)
    
    return f


