#!/usr/bin/python 
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import random

mo=3
P_Con_EE=.5;
EE_Adjlist=[[0]*2]*mo

print random.random();
print random.random();
#for i in EE_Adjlist:
#	print i
	

print EE_Adjlist[2]

for i in range(mo):
	Current_Edge_List=EE_Adjlist[i]
	for j in range(mo):
		if( random.random()< .5):
			#Current_Edge_List=EE_Adjlist[i]
			#In_List_Check= any( elem==j for elem in Current_Edge_List)
			#if( not In_List_Check):
			#	EE_Adjlist[i].append(j)
			#	EE_Adjlist[j].append(i)
			if j in Current_Edge_List:
				pass
			else: 
				EE_Adjlist[i].append(j)
                                EE_Adjlist[j].append(i)
 

for i in EE_Adjlist:
	print i




#	adjlist.append
