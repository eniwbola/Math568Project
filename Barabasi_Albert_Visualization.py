#!/usr/bin/python3.5

#!/usr/bin/python3.7
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import random
import math







def ScaleFree(n,P_Con_EE = 40/(90*2),mo=10,m=2 ): # set P_Con_EE so that average degree is 4

		def find_nearest(array, value):
	    		array = np.asarray(array)
	    		idx = (np.abs(array - value)).argmin()
	    		return idx
		#mo=10

		#EE_Adjlist=[[0]*2]*mo
		EE_Adjlist=[]
		for i in range(mo):
			EE_Adjlist.append([-1]*random.randint(0,2))
		
		angspace=[]
		ypos=[]
		xpos=[]
		xpospre=[]
		ypospre=[]
		for g in range(mo):
			angspace.append((g*2.0*math.pi)/(n*1.0))
		#for h in range(n):
			xpos.append(math.cos(angspace[g]))
			ypos.append(math.sin(angspace[g]))
			xpospre.append(math.cos(angspace[g]))
			ypospre.append(math.sin(angspace[g]))
			ax1=plt1.figure(23)
			plt1.plot(xpospre,ypospre,'ro')
			plt1.xlim([-1.5,1.5])
			plt1.ylim([-1.5,1.5])
			plt1.show(block=False)
			plt1.title('Scale Free via Preferential Attachment')
			plt1.axis('off')
			plt1.pause(0.1)
			
			#plt1.axes.Axes.set_xlim(-1,1)
			#plt1.axes.Axes.set_ylim(-1,1)


		for q in range(40):			
			[i,j] = random.sample(range(mo),2)
			while j in EE_Adjlist[i] :
				[i,j] = random.sample(range(mo),2)
			EE_Adjlist[i].insert(0,j)


		#angspace=[]
		#ypos=[]
		#xpos=[]
		#xpospre=[]
		#ypospre=[]
		xoset=[0,1]
		yoset=[0,1]
		for g in range(mo):
			##angspace.append((g*2.0*math.pi)/(n*1.0))
		#for h in range(n):
			##xpos.append(math.cos(angspace[g]))
			##ypos.append(math.sin(angspace[g]))
			##xpospre.append(math.cos(angspace[g]))
			##ypospre.append(math.sin(angspace[g]))
			
			#xoset=[g]*len(EE_Adjlist[g])
			#yoset=EE_Adjlist[g]
			ax1=plt1.figure(23)
			for gg in range(len(EE_Adjlist[g])):
				xoset[0]=math.cos(angspace[g])
				yoset[0]=math.sin(angspace[g])
				#print(gg)
				#print(EE_Adjlist[g])
				#print(EE_Adjlist[g][gg])
				#print(math.cos(angspace[EE_Adjlist[g][gg] ]))
				#print(len(angspace))
				xoset[1]=math.cos(angspace[EE_Adjlist[g][gg] ])	
				yoset[1]=math.sin(angspace[EE_Adjlist[g][gg] ])
				plt1.plot(xoset,yoset,'r-v')
				
			#plt1.plot(xset,yset,'k-v')
			#plt1.plot(xpospre,ypospre,'ro')
			#plt1.xlim([-1.5,1.5])
			#plt1.ylim([-1.5,1.5])
			#plt1.show(block=False)
			plt1.pause(0.2)





		'''
		for i in range(mo):
			Current_Edge_List=EE_Adjlist[i]
			for j in range(mo):

				if( (random.random()<P_Con_EE) and j!=i):
					if j in Current_Edge_List:
						pass
					else:
						EE_Adjlist[i].insert(0,j)
						EE_Adjlist[j].insert(0,i)
		#for i in EE_Adjlist:		
			#print(i)
		'''
		#print("removing negatives")

		for i in range(len(EE_Adjlist)):
			negcount= EE_Adjlist[i].count(-1)
			for j in range(negcount):
				EE_Adjlist[i].remove(-1)	



		EE_Adj_Count=[]

		for i in EE_Adjlist:
			EE_Adj_Count.append(len(i))


		"""
		for i in range(n-mo):
			EE_Adjlist.append([])
			EE_Adj_Count.append(0)
			for j in range(m):
				if j==0:
					Current_Edge_moi=EE_Adjlist[mo+i]
					Current_Edge_Connect_to=[] # to prevent our mo+i connecting here twice
				Connection_Range=random.uniform(0,1)*sum(EE_Adj_Count)
				#print('Connection_Range', Connection_Range)
				Connect_to=find_nearest(np.cumsum(EE_Adj_Count), Connection_Range)	
				#print('Connect_to' , Connect_to)
			        #if randcol in Current_Edge_List:
			        #pass
				node_add_to_moi=random.randint(0,mo+i-1)
				if j==1 and ( ( Connect_to in Current_Edge_moi)	 or (mo+i in Current_Edge_Connect_to)   ):
                #if j==1 and ( ( Connect_to in Current_Edge_moi)	 or (mo+i in EE_Adjlist[connect_to]) or (node_add_to_moi in Current_Edge_Connect_to)  ):

					j=j-1
					pass	
				EE_Adjlist[mo+i].insert(0,node_add_to_moi)#EE_Adjlist[mo+i].insert(0,k)
				EE_Adjlist[Connect_to].insert(0,mo+i) #EE_Adjlist[k].insert(0,mo+i)
				EE_Adj_Count[mo+i]=EE_Adj_Count[mo+i]+1
				EE_Adj_Count[Connect_to]=EE_Adj_Count[Connect_to]+1
				Current_Edge_moi=EE_Adjlist[mo+i]
				Current_Edge_Connect_to=EE_Adjlist[Connect_to]	
		"""
		
		xset=[0,1]
		yset=[0,1]
		for i in range(n-mo):
			EE_Adjlist.append([])
			EE_Adj_Count.append(0)
			j=0
			while(j<m):
				if j==0:
					Current_Edge_moi=EE_Adjlist[mo+i]
					Current_Edge_Connect_to=[] # to prevent our mo+i connecting here twice
				Connection_Range=random.uniform(0,1)*sum(EE_Adj_Count)
				#print('Connection_Range', Connection_Range)
				Connect_to=find_nearest(np.cumsum(EE_Adj_Count), Connection_Range)	
				#print('Connect_to' , Connect_to)
			        #if randcol in Current_Edge_List:
			        #pass
				node_add_to_moi=random.randint(0,mo+i-1)
				#if j==1 and ( ( Connect_to in Current_Edge_moi)	 or (mo+i in Current_Edge_Connect_to)   ):
				if j==1 and ( ( Connect_to == Current_Edge_Connect_to)	 or (node_add_to_moi in Current_Edge_moi)  ):

					
					continue	
				EE_Adjlist[mo+i].insert(0,node_add_to_moi)#EE_Adjlist[mo+i].insert(0,k)
				EE_Adjlist[Connect_to].insert(0,mo+i) #EE_Adjlist[k].insert(0,mo+i)
				##############################################################

				
		#for h in range(n):
				if j==0:
					angspace.append(((mo+i)*2.0*math.pi)/((n)*1.0))
					xpos.append(math.cos(angspace[mo+i]))
					ypos.append(math.sin(angspace[mo+i]))
				xset[0]=xpos[mo+i]
				yset[0]=ypos[mo+i]
				#if connect<mo
				xset[1]=xpos[Connect_to]
				yset[1]=ypos[Connect_to]
				
				plt1.figure(23)
				plt1.plot(xpos[mo:len(xpos)],ypos[mo:len(ypos)],'ko')
				plt1.plot(xset,yset,'k-v')
				plt1.xlim([-1.5,1.5])
				plt1.ylim([-1.5,1.5])
				plt1.show(block=False)
				plt1.pause(0.01)
                                ###################################################################
				EE_Adj_Count[mo+i]=EE_Adj_Count[mo+i]+1
				EE_Adj_Count[Connect_to]=EE_Adj_Count[Connect_to]+1
				Current_Edge_moi=EE_Adjlist[mo+i]
				Current_Edge_Connect_to=Connect_to
				j=j+1

		plt1.pause(3)
		#print(EE_Adj_Count)	
		EE_Adjlist_Set={}
		Set_Iter=0
		for i in EE_Adjlist:
			EE_Adjlist_Set[Set_Iter]=i
			Set_Iter=Set_Iter+1

		#print(np.cumsum(EE_Adj_Count))
		#print('nearest', find_nearest(np.cumsum(EE_Adj_Count),160))
        	
		return EE_Adjlist_Set

print(ScaleFree(100))
#------------------------------------Aditional Comments----------------------------------------#

#EE_Adjlist,EE_Adj_Count=ScaleFree()  #	adjlist.append
#print "ayyeee"
#for i in EE_Adjlist:
#	print i



#----------------------AlternativeGeneration------------------------------------------#
# for i in range(n):
		# 	totaldeg=sum(EE_Adj_Count)
		# 	EE_Adjlist.append([])
		# 	#EE_Adj_Count.insert(len(EE_Adj_Count),0)
		# 	EE_Adj_Count.append(0)
		# 	for k in range(len(EE_Adjlist)):
		# 		if (k<(mo+i)):
		# 			if random.random()<(len(EE_Adjlist[k])/(sum(EE_Adj_Count)*1.0 ) ):

		# 				for two_it in range(m)
		# 					Connection_Range=random.uniform(0,1)*sum(EE_Adj_Count)
		# 					find_nearest(array, value)
		# 					EE_Adjlist[mo+i].insert(0,random.randint(0,mo+i))#EE_Adjlist[mo+i].insert(0,k)
		# 					EE_Adjlist[k] #EE_Adjlist[k].insert(0,mo+i)
		# 					EE_Adj_Count[mo+i]=EE_Adj_Count[mo+i]+1
		# 					EE_Adj_Count[k]=EE_Adj_Count[k]+1



