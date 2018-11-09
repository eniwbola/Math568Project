#!/usr/bin/python3.5


#!/usr/bin/python2.7




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import random
import math
import networkx as nx
def smallworld(n,P_EE,out_RAD_EE,rp=0.5):
#n=300
#P_Con_EE=.5;
#EE_Adjlist=[[0]*2]*mo
    EE_Adjlist=[]
    for i in range(n):
        EE_Adjlist.append([]) #EE_Adjlist.append([-1]*random.randint(0,2))
	#EE_Adjlist.append(list(map(int,random.randint(0,2))))

    #P_EE=.5

	#out_RAD_EE=3

    for i in range(n):
		
        Current_Edge_List=EE_Adjlist[i]
        for j in range(n):
            dx=math.fabs(i-j)
            if ( (n-dx) < dx):
                dx=n-dx
            if ((dx<=out_RAD_EE) and ( random.random()<=P_EE ) and j!=i  ):
				  
				#Current_Edge_List=EE_Adjlist[i]
                if j in Current_Edge_List:
                    pass
                else:
                    EE_Adjlist[i].insert(0,j)




    for i in range(len(EE_Adjlist)):
        negcount= EE_Adjlist[i].count(-1)
        #print "negcout", negcount
        for j in range(negcount):
            EE_Adjlist[i].remove(-1)	
#----------------------------MatrixGenerationBeforeRewired---------------------#
    EE_Matrix=[[] for i in range(n)]
    print(EE_Matrix)
    for i in range(n):
        for j in range(n):
            if j in EE_Adjlist[i]:
                EE_Matrix[i].append(1)
            else:
                EE_Matrix[i].append(0)
    origmat=np.matrix(EE_Matrix)
    print(origmat)
    
    plt1.matshow(origmat)
    plt1.show()
     #plt1.plot()
    #plt1.xlabel("t")
#----------------------------#--------RewiringProbability--------#-------------------------------------#


# so say we'll look through each list and if the rp<random.random() then we will say choose another node and random between 1 and n
# if that node is not in the list already then we should add that and then pop the previous one 

#    for i in range(len(EE_Adjlist)):
#        if EE_Adjlist[i]:
#            blist=[]
#            for b in EE_Adjlist[i]:
#                if random.uniform(0,1) < rp:
#                    done=0
#                    while done==0:
#                        newloc=random.randint(0,n-1)
#                        if b not in EE_Adjlist[newloc]: 
#                            done=1
#                        else:
#                            done=0
#                    EE_Adjlist[newloc].insert(0,b)
#                    blist.append(b)
#            for b in blist:
#                EE_Adjlist[i].remove(b)

    removed_node_list=[]
    
    for i in range(len(EE_Adjlist)):
        local_removed_node_list=[]
        if EE_Adjlist[i]:
            
            #rewirecounter=0
            for b in EE_Adjlist[i]:
                if random.uniform(0,1) < rp: 
                    removed_node_list.append(b)
                    local_removed_node_list.append(b)
                    #rewirecounter+=1
            for c in local_removed_node_list:
                EE_Adjlist[i].remove(c) 
                       
                   
    for a in range(len(removed_node_list)):
        newloc=random.randint(0,n-1)
        done=0
        
        while done==0:
            if (removed_node_list[a] not in EE_Adjlist[newloc]) and (newloc != removed_node_list[a]):
                done=1
                EE_Adjlist[newloc].insert(0,removed_node_list[a])
            else:
                done=0
                newloc=random.randint(0,n-1)
 #--------------------matrix genration after rewired\-----------------------#                     
    EE_Matrix=[[] for i in range(n)]
    print(EE_Matrix)
    for i in range(n):
        for j in range(n):
            if j in EE_Adjlist[i]:
                EE_Matrix[i].append(1)
            else:
                EE_Matrix[i].append(0)
    newmat=np.matrix(EE_Matrix)
    print(origmat)
    
    plt1.matshow(newmat)
    plt1.show()

#-----------------------------------------------------------------#
    EE_Adj_Count=[]

    EE_Adjlist_Set={}

    Set_Iter=0

    for i in EE_Adjlist:
        EE_Adjlist_Set[Set_Iter]=i
        Set_Iter=Set_Iter+1


    return(EE_Adjlist_Set)    

	#	adjlist.append

#smallworld(300,1,12,1)
#smallworld(n,P_EE,out_RAD_EE,rp):


