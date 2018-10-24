from pylab import plot, xlabel, show, legend
from numpy import array, arange
from math import exp

def HH(x):

    m = x[0]
    h = x[1]
    n = x[2]
    v = x[3]

    a_m = ((v + 45)/10)/(1-exp(-(v+45)/10))
    b_m = 4*exp(-(v+70)/18)

    a_h = 0.07*exp(-(v+70)/20)
    b_h = 1/(exp(-(v+40)/10)+1)

    a_n = 0.01*(v+60)/(1-exp(-(v+60)/10))
    b_n = exp(-(v+70)/80) /8

    fm = a_m * (1 - m) - b_m * m
    fh = a_h * (1 - h) - b_h * h
    fn = a_n * (1 - n) - b_n * n

    C = 1 #uf/cm2
    v_Na = 45 #mV
    v_K = -82 #mV
    v_L = -59 #mV
    g_Na = 120 #mS/cm2
    g_K = 36 #mS/cm2
    g_L = 0.3 #mS/cm2
    I = 10 #uA/cm2


    fv = (1/C)*(g_Na*(m**3)*h*(v_Na-v)+g_K*(n**4)*(v_K-v)+g_L*(v_L-v)+I)

    return array([fm,fh,fn,fv],float)

#steps setup
a = 0.0
b = 100.0
delta_t = 0.01 #delta_t = 0.01 ms
N = (b-a)/delta_t


tpoints = arange(a,b,delta_t)
vpoints = []
mpoints = []
npoints = []
hpoints = []



x = array([0.0,0.0,0.0,-70],float) #setup the initial values for m,h,n

for t in tpoints:
    vpoints.append(x[3])
    mpoints.append(x[0])
    hpoints.append(x[1])
    npoints.append(x[2])
    k1 = delta_t * HH(x)
    k2 = delta_t * HH(x + 0.5 * k1)
    k3 = delta_t * HH(x + 0.5 * k2)
    k4 = delta_t * HH(x + k3)

    x += (k1 + 2*k2 + 2*k3 +k4)/6

plot(tpoints,vpoints)
xlabel("t")
show()
plot(tpoints,mpoints)
plot(tpoints,hpoints)
plot(tpoints,npoints)
#legend()
xlabel("t")
show()