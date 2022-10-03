# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 18:55:17 2021

@author: e.christofi
"""

import mdtraj as md
import numpy as np
from copo import load_chemistry
from params import *
import random
import pickle

counter = 0


#load the chemical structure of the system
chemistry_filename = '../Data/chem_tcv_451045.txt'
chemistry, chemistry_names = load_chemistry(chemistry_filename, None)
chemistry = np.array(chemistry)
print("Number of Chains per configuration:",chemistry.shape[0])
print("Number of Monomers per chain:",chemistry.shape[1])

#load the trajectory
t = md.load('../Data/trajout_long.gro')
print(t)

#compute the number of particles per chain
def calc_chainlens(chemistry,merlen_cis,merlen_trans,merlen_vinyl):
    chainlens = []
    for i in range(chemistry.shape[0]):
      ntypes = [np.count_nonzero(chemistry[i] == 0),np.count_nonzero(chemistry[i] == 1),np.count_nonzero(chemistry[i] == 2)] 
      chainlen = ntypes[0]*merlen_cis + ntypes[1]*merlen_trans + ntypes[2]*merlen_vinyl
      chainlens.append(int(chainlen))
    
    chainlens = np.array(chainlens, dtype=np.uint8)
    np.savetxt("./chainlens.txt",chainlens)
    return chainlens  

chainlens = calc_chainlens(chemistry,merlen_cis,merlen_trans,merlen_vinyl)



#compute the bond vectors for each monomer type
def cis_vectors(index,imageAT,Coord,jj,b0):
    bv1=Coord[1+index]-Coord[index]
    bv2=Coord[2+index]-Coord[1+index]
    bv3=Coord[3+index]-Coord[2+index]
    if jj != nmer-1:
      bv0=Coord[4+index]-Coord[3+index]
    else:
      bv0= b0[:]
    imageAT.append([bv1[0],bv1[1],bv1[2]])
    imageAT.append([bv2[0],bv2[1],bv2[2]])
    imageAT.append([bv3[0],bv3[1],bv3[2]])      
    imageAT.append([bv0[0],bv0[1],bv0[2]])
    return bv0     

def trans_vectors(index,imageAT,Coord,jj,b0):
    bv1=Coord[1+index]-Coord[index]
    bv2=Coord[2+index]-Coord[1+index]
    bv3=Coord[3+index]-Coord[2+index]
    if jj != nmer-1:
      bv0=Coord[4+index]-Coord[3+index]
    else:
      bv0= b0[:]
    imageAT.append([bv1[0],bv1[1],bv1[2]])
    imageAT.append([bv2[0],bv2[1],bv2[2]])
    imageAT.append([bv3[0],bv3[1],bv3[2]])      
    imageAT.append([bv0[0],bv0[1],bv0[2]])
    return bv0     

def vinyl_vectors(index,imageAT,Coord,jj,b0):
    bv1=Coord[1+index]-Coord[index]
    bv2=Coord[2+index]-Coord[1+index]
    bv3=Coord[3+index]-Coord[2+index]
    if jj != nmer-1:
      bv0=Coord[4+index]-Coord[1+index]
    else:
      bv0= b0[:]    
    imageAT.append([bv1[0],bv1[1],bv1[2]])
    imageAT.append([bv2[0],bv2[1],bv2[2]])
    imageAT.append([bv3[0],bv3[1],bv3[2]])      
    imageAT.append([bv0[0],bv0[1],bv0[2]])
    return bv0     




def encoding(frames,save_path):  
 input_file=[]
 target_file=[]  
 b0 = np.zeros((3))
 for frameIndx in frames:
    # Counter to ignore the last particle of the chain, when the chain ends in vinly-1,2
    counter=0
    
    # Compute the length of the box
    LXX,LYY,LZZ=t.unitcell_lengths[frameIndx][0],t.unitcell_lengths[frameIndx][1],t.unitcell_lengths[frameIndx][2]
    
    hLXX,hLYY,hLZZ=LXX/2.0,LYY/2.0,LZZ/2.0
    Coord=np.zeros([npart,3],dtype=np.float32)
    
    # Counter for the particles of the frame
    frame_counter_1 = -1
    
    frame_counter_2 = 0
    for j in range(nchain):
        coordCG=np.zeros([nmer,3],dtype=np.float32)
        coordAT = []  
        for jj in range(nmer):
            # Compute the type of the monomer
            type = chemistry[j][jj]
            
            posCM=[0.,0.,0.]
            if(type==0): masses,merlen = masses_cis,merlen_cis
            elif(type==1): masses,merlen = masses_trans,merlen_trans
            elif(type==2): masses,merlen = masses_vinyl,merlen_vinyl
            totmass=np.sum(masses)
            for ii in range(merlen):
                frame_counter_1 += 1
                partIndx = frame_counter_1
                
                # Compute the un-wrapped coordinates of the particles
                Coord[partIndx]=t.xyz[frameIndx,(partIndx+counter),:]
                if (jj!=0 or ii!=0):
                    if(Coord[partIndx][0]-Coord[partIndx-1][0]<-hLXX): Coord[partIndx][0]+=LXX
                    if(Coord[partIndx][0]-Coord[partIndx-1][0]>hLXX): Coord[partIndx][0]-=LXX
                    if(Coord[partIndx][1]-Coord[partIndx-1][1]<-hLYY): Coord[partIndx][1]+=LYY
                    if(Coord[partIndx][1]-Coord[partIndx-1][1]>hLYY): Coord[partIndx][1]-=LYY
                    if(Coord[partIndx][2]-Coord[partIndx-1][2]<-hLZZ): Coord[partIndx][2]+=LZZ
                    if(Coord[partIndx][2]-Coord[partIndx-1][2]>hLZZ): Coord[partIndx][2]-=LZZ
           
            # Compute the coordinates of the CG particles of the chain 
                posCM[0]+=Coord[partIndx][0]*masses[ii]
                posCM[1]+=Coord[partIndx][1]*masses[ii]
                posCM[2]+=Coord[partIndx][2]*masses[ii]

            posCM[0]/=totmass
            posCM[1]/=totmass
            posCM[2]/=totmass
            # Save the coordinates of the CG particles of the chain 
            coordCG[jj]=posCM[:]
            
        # Ignore the last particle of the chain, when the chain ends in vinyl-1,2    
        if(j in cv3): counter+=1
        
        for jj in range(nmer):
            type = chemistry[j][jj]
            
            # Compute the bond vectors for each monomer 
            if(type==0):
                merlen = merlen_cis
                b0[:]=cis_vectors(frame_counter_2,coordAT,Coord,jj,b0)
            elif(type==1): 
                merlen = merlen_trans
                b0[:]=trans_vectors(frame_counter_2,coordAT,Coord,jj,b0)
            elif(type==2):
                merlen = merlen_vinyl  
                b0[:]=vinyl_vectors(frame_counter_2,coordAT,Coord,jj,b0)

            frame_counter_2 += merlen

        coordAT = np.array(coordAT,dtype=np.float64)                
        
        # Compute the number of splits (samples) per chain 
        nsplits = int(np.ceil((coordAT.shape[0])/INPUT_WIDTH))  
        s = int(np.ceil(nmer/nsplits)) 
        if (s*(4)) > INPUT_WIDTH: nsplits += 1
        s = int(np.ceil(nmer/nsplits))
        meridx = [(s*(h+1)-1) for h in range(nsplits)]
        meridx.pop(-1)
        meridx.append(nmer-1)
        AT = []
        CG = []
        frame_counter = -1
        arrAT = np.zeros([INPUT_WIDTH,3],dtype=np.float64)
        arrCG = np.zeros([INPUT_WIDTH,6],dtype=np.float64)
        for jj in range(nmer):
            type = chemistry[j][jj]
            if(type==0): 
                dtseg = dtseg_cis
                m_id = [0,0,1]
            elif(type==1):
                dtseg = dtseg_trans
                m_id = [0,1,0]
            elif(type==2): 
                dtseg = dtseg_vinyl
                m_id = [1,0,0]
            for g in range(dtseg):
                frame_counter += 1
        
                AT.append([coordAT[frame_counter][0],coordAT[frame_counter][1],coordAT[frame_counter][2]])
                CG.append([coordCG[jj][0],coordCG[jj][1],coordCG[jj][2],m_id[0],m_id[1],m_id[2]])
            
            # Create input and target samples 
            if jj in meridx: 
                AT = np.array(AT,dtype=np.float64)
                CG = np.array(CG,dtype=np.float64)
                
                # Compute the zero-padding 
                shiftx = int(np.ceil((INPUT_WIDTH - AT.shape[0])/2))
                
                arrAT[shiftx:shiftx+int(AT.shape[0])] = AT[:]
                arrCG[shiftx:shiftx+int(CG.shape[0])] = CG[:]  
                input_file.append(arrCG)
                target_file.append(arrAT)                
                AT = []
                CG = []
                arrAT = np.zeros([INPUT_WIDTH,3],dtype=np.float64)
                arrCG = np.zeros([INPUT_WIDTH,6],dtype=np.float64)
                
 final_input=np.array(input_file,dtype=np.float64)
 final_target=np.array(target_file,dtype=np.float64) 
 print(final_input.shape, final_target.shape)      
 
 with open(save_path+'_input.pkl','wb') as f:
     pickle.dump(final_input, f)

 with open(save_path+'_target.pkl','wb') as f:
     pickle.dump(final_target, f)        
 input_file = []
 target_file = []



random.seed(100)
frames = random.sample(range(1851), 1851)

save_path="train" 
encoding(frames[:1481],save_path)

save_path="val" 
encoding(frames[1481:1666],save_path) 

save_path="test" 
encoding(frames[1666:],save_path) 
print(frames[1666:])
