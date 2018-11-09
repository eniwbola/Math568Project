
# generate the random graph----
from random import sample
from math import log
def RandomG(n,c):
    #n = 100
    #c = 2*log(2)
    m = round(n*c)
    adlist = {}
    #degrees = []
    for i in range(n):
        adlist[i] = []
    for i in range(m):
        [pre,post] = sample(range(n),2)   
        adlist[post].append(pre)        
    # ----------------------------------
    #print(adlist)
    return adlist