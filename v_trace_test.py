from pylab import plot, xlabel, show, legend
from numpy import array, arange
from math import exp
from Cortical_neuron import Cortical_neuron
#steps setup
a = 0.0
b = 1000.0
delta_t = 0.01 #delta_t = 0.01 ms
I = 1.3
N = (b-a)/delta_t


tpoints = arange(a,b,delta_t)
vpoints = []
zpoints = []
npoints = []
hpoints = []



x = array([0.0,0.0,0.0,-70],float) #setup the initial values for m,h,n

for t in tpoints:
    vpoints.append(x[3])
    zpoints.append(x[0])
    hpoints.append(x[1])
    npoints.append(x[2])
    k1 = delta_t * Cortical_neuron(x,I)
    k2 = delta_t * Cortical_neuron(x + 0.5 * k1,I)
    k3 = delta_t * Cortical_neuron(x + 0.5 * k2,I)
    k4 = delta_t * Cortical_neuron(x + k3,I)

    x += (k1 + 2*k2 + 2*k3 +k4)/6

plot(tpoints,vpoints)
xlabel("t")
show()
plot(tpoints,zpoints)
plot(tpoints,hpoints)
plot(tpoints,npoints)
#legend()
xlabel("t")
show()