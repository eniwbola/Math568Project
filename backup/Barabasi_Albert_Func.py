#!/usr/bin/python 
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import random






def ScaleFree():
	mo=10
	P_Con_EE=.5;
	#EE_Adjlist=[[0]*2]*mo
	EE_Adjlist=[]
	for i in range(mo):
		EE_Adjlist.append([-1]*random.randint(0,2))
		#EE_Adjlist.append(list(map(int,random.randint(0,2))))
	print random.random();
	print random.random();
	#for i in EE_Adjlist:
	#	print i
	print "yeeeeeeeeeeeeee"	

	print EE_Adjlist[2]

	for i in range(mo):
		Current_Edge_List=EE_Adjlist[i]
		for j in range(mo):
			if( (random.random()< .5) and j!=i):
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
	#	print i
		EE_Adj_Count.append(len(i))

	#for i in EE_Adj_Count:
	#	print i



	for i in range(3):
		totaldeg=sum(EE_Adj_Count)
		EE_Adjlist.append([])
		#EE_Adj_Count.insert(len(EE_Adj_Count),0)
		EE_Adj_Count.append(0)
		for k in range(len(EE_Adjlist)):
			if (k<(mo+i)):
				if random.random()<(len(EE_Adjlist[k])/(sum(EE_Adj_Count)*1.0 ) ):
					EE_Adjlist[mo+i].insert(0,k)
					EE_Adjlist[k].insert(0,mo+i)
					EE_Adj_Count[mo+i]=EE_Adj_Count[mo+i]+1
					EE_Adj_Count[k]=EE_Adj_Count[k]+1





	#for i in EE_Adjlist:
	#	print i 
				
	
	#for i in EE_Adj_Count:
	#	print i 
	
	return EE_Adjlist, EE_Adj_Count	

EE_Adjlist,EE_Adj_Count=ScaleFree()  #	adjlist.append
print "ayyeee"
for i in EE_Adjlist:
	print i


