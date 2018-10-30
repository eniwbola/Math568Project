#!/usr/bin/python2.7




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import random
import math
#import networkx as nx
n=300
P_Con_EE=.5;
#EE_Adjlist=[[0]*2]*mo
EE_Adjlist=[]
for i in range(n):
	EE_Adjlist.append([-1]*random.randint(0,2))
	#EE_Adjlist.append(list(map(int,random.randint(0,2))))

P_EE=.5

out_RAD_EE=3

for i in range(n):
	
	Current_Edge_List=EE_Adjlist[i]
	for j in range(n):
		dx=math.fabs(i-j)
		if ( (n-dx) < dx):
			dx=n-dx
		if ((dx<out_RAD_EE) and ( random.random()<=P_EE ) and j!=i  ):
			  
			#Current_Edge_List=EE_Adjlist[i]
			if j in Current_Edge_List:
				pass
			else:
				EE_Adjlist[i].insert(0,j)
                                EE_Adjlist[j].insert(0,i)
#EE_Adjlist.append([])
for i in EE_Adjlist:		
	print i

print "removing negatives"

for i in range(len(EE_Adjlist)):
	negcount= EE_Adjlist[i].count(-1)
	#print "negcout", negcount
	for j in range(negcount):
		EE_Adjlist[i].pop(-1)	



EE_Adj_Count=[]

for i in EE_Adjlist:
	print i
	EE_Adj_Count.append(len(i))

for i in EE_Adj_Count:
	print i









for i in EE_Adjlist:
	print i 
				
	
for i in EE_Adj_Count:
	print i 
	
	

#	adjlist.append
