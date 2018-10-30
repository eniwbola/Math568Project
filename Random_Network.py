#!/usr/bin/python2.7

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import random
import math
#import networkx as nx
# so when this becomes a function will have you put in Excitatory Number, Inhibitory Number, was well as the probabilities pee,pii,pie,pei and it will spit out 1 to 4 matrices that determine the connections, the rest is up to you, be brave my friend. 
nee=300;
pee=.05;#c=2*(math.log(2))
#m= round((c*n/2))#30;
print m ,'m'
#P_Con_EE=.5;
#EE_Adjlist=[[0]*2]*mo
Edge_Tot=m
EE_Adjlist=[]
for i in range(n):
	EE_Adjlist.append([-1]*random.randint(0,2))
	#EE_Adjlist.append(list(map(int,random.randint(0,2))))

P_EE=.5

out_RAD_EE=3


for i in EE_Adjlist:
        print i

Current_Edge_List=[]

while Edge_Tot>0:
	randrow=random.randint(1,n)-1
	randcol=random.randint(1,n)-1
	if ((Edge_Tot>0) and (randrow!=randcol)):
		Current_Edge_List=EE_Adjlist[randrow];
		if randcol in Current_Edge_List:
			pass
		else:
			#print randcol
			#print randrow
			#print len(EE_Adjlist)
			EE_Adjlist[randrow].insert(0,randcol)
			
			EE_Adjlist[randcol].insert(0,randrow)
			Edge_Tot=Edge_Tot-1 
#for i in range(n):
#	
#	Current_Edge_List=EE_Adjlist[i]
#	for j in range(n):
#		dx=math.fabs(i-j)
#		if ( (n-dx) < dx):
#			dx=n-dx
#		if ((dx<out_RAD_EE) and ( random.random()<=P_EE ) and j!=i  ):
			  
			#Current_Edge_List=EE_Adjlist[i]
#			if j in Current_Edge_List:
#				pass
#			else:
#				EE_Adjlist[i].insert(0,j)
 #                               EE_Adjlist[j].insert(0,i)



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
	
print sum(EE_Adj_Count)	

#	adjlist.append


Comp_Size=0;

for i in range(n):
	Buffer_Array=np.zeros(n)
	Distance_Array=np.ones(n)
	Size_Array=np.ones(n)
	Read_Iter=0;
	Write_Iter=1;

	The_First_Test_Node=22961

	for b in range(n):
		Distance_Array[b]=-1



	#Distance_Array[ The_First_Test_Node ]=0
	#Buffer_Array[0]=The_First_Test_Node
	Distance_Array[n-1]=0
	Buffer_Array[0]=n-1


	while Read_Iter!=Write_Iter:
		Reading=int(Buffer_Array[Read_Iter])
		#print Reading
		Read_Iter=Read_Iter+1

		Distance_D=Distance_Array[Reading]
		for m in EE_Adjlist[Reading]:
			#print adjlist[Reading]
			if Distance_Array[m]!=-1:
				pass
			else:
				Distance_Array[m]=Distance_D+1
				Buffer_Array[Write_Iter]=m	
				Write_Iter=Write_Iter+1
			#print Read_Iter,Write_Iter
	if Read_Iter>Comp_Size:
		Comp_Size=Read_Iter
	#print "Comp_Size", Comp_Size		
	
#for b in Distance_Array:
	#print b #Distance_Array
print "Comp_Size", Read_Iter
print "Closeness Centrality", 1/(np.mean(Distance_Array))






