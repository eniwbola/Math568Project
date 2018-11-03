#!/usr/bin/python2.7




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import random
import math
import Small_World
import Barabasi_Albert
import networkx as nx


#Small_World.smallworld(300,.5)
#sw_network=Small_World.smallworld(300,.5)
#g=nx.DiGraph(sw_network)
Scale_Free_Network=Barabasi_Albert.ScaleFree(15,.5,10,2)
print 'scalin'
print Scale_Free_Network

g=nx.DiGraph(Scale_Free_Network)


pos=nx.spring_layout(g)
nx.draw_networkx_nodes(g,pos) 
nx.draw_networkx_edges(g,pos) 
nx.draw(g,pos, with_labels=False)

plt1.show()