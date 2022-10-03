# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:38:50 2021

@author: e.christofi
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from collections import defaultdict
import time
from params import *
import os

#mute the warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 16})

# Calculate bond lengths
def dist(p1,p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)**0.5

# Calculate bond angles
def angle(p1,p2,p3):
    vec1=[p1[0]-p2[0],p1[1]-p2[1],p1[2]-p2[2]]
    vec2=[p3[0]-p2[0],p3[1]-p2[1],p3[2]-p2[2]]
    return np.arccos(np.dot(vec1,vec2)/np.sqrt(np.dot(vec1,vec1))/np.sqrt(np.dot(vec2,vec2)))/np.pi*180.0

# Calculate dihedral angles
def dihed(p1,p2,p3,p4):
    b0=[p1[0]-p2[0],p1[1]-p2[1],p1[2]-p2[2]]
    b1=[p3[0]-p2[0],p3[1]-p2[1],p3[2]-p2[2]]
    b2=[p4[0]-p3[0],p4[1]-p3[1],p4[2]-p3[2]]

    b1/=np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))



cis = ['CC','CDC','CDC','CC'] 
trans = ['CT','CDT','CDT','CT']
vinyl = ['CV2','CV1','CDV1','CDV2']   
types = [cis,trans,vinyl] 
comb = []

#comb contains all the possible combinations between the monomers
for i in range(len(types)):
  for j in range(len(types)):
     for z in range(len(types)): 
      s = types[i] + types[j] + types[z]
      comb.append(s)

#compute the bond length for each combination of two atoms for the target and predicted coordinates
def calculate_bond_lengths(comb,data):     
  bond_lengths = defaultdict(list)
    
  for j in range(nchain):  
   for w in range(chainlen-1):
    i = j*chainlen + w    
    name_list = [data[i+e][0].astype(str) for e in range(2)]  
    if name_list[0] == "CDV2":
         name = data[i-2][0].astype(str)+"-"+name_list[1]
         bond_lengths[name].append(dist(data[i-2][1:].astype(float), data[i+1][1:].astype(float)))   
    else:     
         name = name_list[0]+"-"+name_list[1]
         bond_lengths[name].append(dist(data[i][1:].astype(float), data[i+1][1:].astype(float)))   
    
  print("Bond lenghts combinations: ",len(bond_lengths)) 
  
  return bond_lengths

#compute the bond angle for each combination of three atoms for the target and predicted coordinates
def calculate_bond_angles(comb,data):     
  bond_angles = defaultdict(list)

  for j in range(nchain):  
   for w in range(chainlen-2):
    i = j*chainlen + w  
    name_list = [data[i+e][0].astype(str) for e in range(3)]
    if name_list[0] == "CDV1":
        name = data[i-2][0].astype(str)+"-"+ data[i-1][0].astype(str)+"-"+name_list[2]
        bond_angles[name].append(angle(data[i-2][1:].astype(float), data[i-1][1:].astype(float), data[i+2][1:].astype(float))) 
        name = data[i][0].astype(str)+"-"+ data[i-1][0].astype(str)+"-"+name_list[2]
        bond_angles[name].append(angle(data[i][1:].astype(float), data[i-1][1:].astype(float), data[i+2][1:].astype(float))) 
    elif name_list[0] == "CDV2":
        name = data[i-2][0].astype(str)+"-"+name_list[1]+"-"+name_list[2]
        bond_angles[name].append(angle(data[i-2][1:].astype(float), data[i+1][1:].astype(float), data[i+2][1:].astype(float)))     
    else:
        name = name_list[0]+"-"+name_list[1]+"-"+name_list[2]
        bond_angles[name].append(angle(data[i][1:].astype(float), data[i+1][1:].astype(float), data[i+2][1:].astype(float)))   
  print("Bond angles combinations: ",len(bond_angles)) 
  return bond_angles

#compute the dihedral angles for each combination of four atoms for the target and predicted coordinates
def calculate_dihedral_angles(comb,data):     
  dihedral_angles = defaultdict(list)

    
  for j in range(nchain):  
   for w in range(chainlen-3):
    i = j*chainlen + w  
    name_list = [data[i+e][0].astype(str) for e in range(4)]
    if (name_list[0] == "CV1"):
      if (data[i-2][0].astype(str)=="CDV2"):
        name = data[i-4][0].astype(str)+"-"+data[i-1][0].astype(str)+"-"+name_list[0]+"-"+name_list[3]
        dihedral_angles[name].append(dihed(data[i-4][1:].astype(float), data[i-1][1:].astype(float), data[i][1:].astype(float),data[i+3][1:].astype(float)))       
      else:    
        name = data[i-2][0].astype(str)+"-"+data[i-1][0].astype(str)+"-"+name_list[0]+"-"+name_list[3]
        dihedral_angles[name].append(dihed(data[i-2][1:].astype(float), data[i-1][1:].astype(float), data[i][1:].astype(float),data[i+3][1:].astype(float))) 
    elif (name_list[0] == "CDV1"):
        name = data[i-2][0].astype(str)+"-"+data[i-1][0].astype(str)+"-"+name_list[2]+"-"+name_list[3]
        dihedral_angles[name].append(dihed(data[i-2][1:].astype(float), data[i-1][1:].astype(float), data[i+2][1:].astype(float),data[i+3][1:].astype(float)))
        name = data[i][0].astype(str)+"-"+data[i-1][0].astype(str)+"-"+name_list[2]+"-"+name_list[3]
        dihedral_angles[name].append(dihed(data[i][1:].astype(float), data[i-1][1:].astype(float), data[i+2][1:].astype(float),data[i+3][1:].astype(float)))
    elif (name_list[0] == "CDV2"):
        name = data[i-2][0].astype(str)+"-"+name_list[1]+"-"+name_list[2]+"-"+name_list[3]
        dihedral_angles[name].append(dihed(data[i-2][1:].astype(float), data[i+1][1:].astype(float), data[i+2][1:].astype(float),data[i+3][1:].astype(float))) 
        name = data[i][0].astype(str)+"-"+data[i-1][0].astype(str)+"-"+data[i-2][0].astype(str)+"-"+data[i+1][0].astype(str)
        dihedral_angles[name].append(dihed(data[i][1:].astype(float), data[i-1][1:].astype(float), data[i-2][1:].astype(float),data[i+1][1:].astype(float))) 
    else:
        name = name_list[0]+"-"+name_list[1]+"-"+name_list[2]+"-"+name_list[3]
        dihedral_angles[name].append(dihed(data[i][1:].astype(float), data[i+1][1:].astype(float), data[i+2][1:].astype(float),data[i+3][1:].astype(float)))   
  print('Dihedral angles combinations: ',len(dihedral_angles)) 
  
  return dihedral_angles


def count_dictionary(dictionary): 
 s = 0
 for i in list(dictionary):
    if len(dictionary[i]) == 0:
        print('key is empty')
    else:
       s = s + len(dictionary[i])
    
 print(s)


os.mkdir("./bond_angles/")
os.mkdir("./bond_lengths/")
os.mkdir("./dihedral_angles/")
os.mkdir("./results/")

for epoch in [1,50,100,200,300,400,500,600,700,800,900,1000]:

 data = './Data_'+str(epoch)
 
 frames = com_list
 
 totalA = 0
 totalB = 0
 totalC = 0
 frame_c = 0

 bl_o = defaultdict(list)
 ba_o = defaultdict(list)
 da_o = defaultdict(list)

 bl_p = defaultdict(list)
 ba_p = defaultdict(list)
 da_p = defaultdict(list)



 for frame in frames:
  start = time.time()   
  #load the data   
  with open(data+'/PStart_'+str(frame)+'.dat') as f1:
     lines = (line for line in f1 if not line.startswith('#'))
     Pdata = np.loadtxt(lines, delimiter=' ',skiprows=2,dtype=(np.string_,np.float_))    
 
  with open(data+'/TStart_'+str(frame)+'.dat') as f1:
     lines = (line for line in f1 if not line.startswith('#'))
     Odata = np.loadtxt(lines, delimiter=' ',skiprows=2,dtype=(np.string_,np.float_))  

  frame_c += 1
  print(frame_c)   

  


  #compute distibutions for the predicted coordinates
  bond_lengths_p = calculate_bond_lengths(comb,Pdata)
  bond_angles_p = calculate_bond_angles(comb,Pdata)
  dihedral_angles_p = calculate_dihedral_angles(comb,Pdata)

  #compute distibutions for the target coordinates
  bond_lengths_o = calculate_bond_lengths(comb,Odata)
  bond_angles_o = calculate_bond_angles(comb,Odata)
  dihedral_angles_o = calculate_dihedral_angles(comb,Odata)  

  for i in bond_lengths_o.keys():
    bl_o[i] += bond_lengths_o[i]
    bl_p[i] += bond_lengths_p[i]
    

  for i in bond_angles_o.keys():
    ba_o[i] += bond_angles_o[i]
    ba_p[i] += bond_angles_p[i]
   

  for i in dihedral_angles_o.keys():
    da_o[i] += dihedral_angles_o[i]
    da_p[i] += dihedral_angles_p[i]
    

  comp_bond_lengths = list(bond_lengths_p)
  sumA = 0
  for i in comp_bond_lengths:
   hist1, bin_edges1 = np.histogram(bond_lengths_o[i],bins=40,range=(0.05,0.25),density=True)
 
   hist3, bin_edges3 = np.histogram(bond_lengths_p[i],bins=40,range=(0.05,0.25),density=True)
   sumA+=(np.linalg.norm((hist1-hist3), ord=1))
  sumA /=3000.  
  totalA += sumA 
 
  sumB = 0
  comp_bond_angles = list(bond_angles_p)
  for i in comp_bond_angles:
   hist1, bin_edges1 = np.histogram(bond_angles_o[i],bins=90,range=(0.0,180.0),density=True)

   hist3, bin_edges3 = np.histogram(bond_angles_p[i],bins=90,range=(0.0,180.0),density=True)
   sumB+=(np.linalg.norm((hist1-hist3), ord=1))
  sumB /=10.5 
  totalB += sumB 
  
  comp_dihedral_angles = list(dihedral_angles_p)
  sumC = 0
  for i in comp_dihedral_angles:
   hist1, bin_edges1 = np.histogram(dihedral_angles_o[i],bins=180,range=(-180,180),density=True)

   hist3, bin_edges3 = np.histogram(dihedral_angles_p[i],bins=180,range=(-180,180),density=True)
   sumC+=(np.linalg.norm((hist1-hist3), ord=1))
  sumC /=13.5 
  totalC += sumC
  print(time.time()-start)
 nframes = len(frames)
 totalA /= nframes 
 totalB /= nframes
 totalC /= nframes
 total_psi = totalA + totalB + totalC

 print(epoch,float(totalA),float(totalB),float(totalC),float(total_psi),file=open("./psi.dat","a+"))

 for i in comp_bond_angles:
    fig,ax = plt.subplots()
    sns.histplot(ba_o[i],bins=90,binrange=(0.0,180.), stat="density",  ax=ax, color="red", label="Target", fill=False,element="poly")
    sns.histplot(ba_p[i],bins=90,binrange=(0.0,180.), stat="density", ax=ax, color="blue", label="Prediction", fill=False,element="poly")
    plt.title(label='Bond angle: '+i)
    plt.xlabel('bond angle (degrees)')
    plt.ylabel('probability density')
    plt.legend(fontsize=14) 
    plt.savefig('./bond_angles/{}.png'.format(i), bbox_inches='tight')
    plt.close()
    
 for i in comp_bond_lengths:
    fig,ax = plt.subplots()
    sns.histplot(bl_o[i],bins=40,binrange=(0.05,0.25), stat="density",  ax=ax, color="red", label="Target", fill=False,element="poly")
    sns.histplot(bl_p[i],bins=40,binrange=(0.05,0.25), stat="density",  ax=ax, color="blue", label="Prediction", fill=False,element="poly")
    plt.title(label='Bond length: '+i)
    plt.xlabel('bond length (nm)')
    plt.ylabel('probability density')
    plt.legend(fontsize=14) 
    plt.savefig('./bond_lengths/{}.png'.format(i), bbox_inches='tight')
    plt.close()  
    
 for i in comp_dihedral_angles:
    fig,ax = plt.subplots()
    sns.histplot(da_o[i],bins=180,binrange=(-180.,180.), stat="density",  ax=ax, color="red", label="Target", fill=False,element="poly")
    sns.histplot(da_p[i],bins=180,binrange=(-180.,180.), stat="density", ax=ax, color="blue", label="Prediction", fill=False,element="poly")
    plt.title(label='Dihedral angle: '+i)
    plt.xlabel('dihedral angle (degrees)')
    plt.ylabel('probability density')
    plt.legend(fontsize=14) 
    plt.savefig('./dihedral_angles/{}.png'.format(i), bbox_inches='tight')
    plt.close()    

 if epoch == 1000:
    psis = np.loadtxt("./psi.dat", unpack=True)
    ylabels = [r"${\Psi}_{bond-length}$", r"${\Psi}_{bond-angle}$", r"${\Psi}_{dihedral-angle}$", r"${\Psi}_{total}$"]
    labels =["bond_lengths", "bond_angles", "dihedral_angles","total_error"]
    for i in range(psis.shape[0]-1):
         fig,ax = plt.subplots()        
         ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
         plt.plot(psis[0,:], psis[i+1,:],  marker='o',label=labels[i])
         plt.ylabel(ylabels[i])
         plt.xlabel("epochs")
         plt.legend() 
         plt.savefig('./results/{}.png'.format(labels[i]))
         plt.close()
         
