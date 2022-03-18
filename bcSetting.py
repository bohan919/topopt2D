# This script sets multiple boundary conditions as well as loading conditions

import numpy as np
import random

# CONDITION SETTING
def main(fnno,stype,weight_factor):
# fnno - number of loads
# stype - bc type

    ar=1
    volfrac=round(np.random.normal(loc=0.4, scale=0.07, size=None),2)
    if volfrac <= 0.3:
        volfrac=0.3
    if volfrac >= 0.5:
        volfrac=0.5
        
#     ar=random.randint(0, 1) # 4 prepared aspect ratio
#     if stype == 3:
#         ar=random.randint(0,1) #The final aspect ratio condition only for stype=3
   
    # at least 16 elements in one certain direction to show clear and meaningful structures when applying force in this direction
    if ar == 0: #AR1
        nelx=64
        nely=16
        nelz=8    
        rmin=1.2 #0.15*length
    # the best one for plotting
    if ar == 1: #AR2
#         nelx=64
#         nely=8            
#         nelz=16
#         rmin=1.2 #0.15*l 
     
# 64 16 16
    #new aspect ratio
        nelx=32
        nely=16
        nelz=16
        rmin=1.2 #0.15*l 
        
        
        
        # not enough data in this direction
    if ar == 2: #AR4
        nelx=32
        nely=8
        nelz=32
        rmin=1.2 #0.15*l
    if ar == 3: #AR5 # only BC 2,3,4
        nelx=16
        nely=32
        nelz=16
        rmin=2.4 #0.15*l
        
    il = np.arange(fnno)
    jl = np.arange(fnno)
    kl = np.arange(fnno)
    fx = np.arange(fnno)
    fy = np.arange(fnno)
    fz = np.arange(fnno)
    case = np.arange(fnno)  # index to judge force aplly on which side
    loadnid = np.arange(fnno)
    
    
    if stype==0: #cantilever beam  #set the potential force domian x: [nelx/2,nelx] y:[0,nely]
  
        for i in range(fnno):        

            il[i] = nelx
            jl[i] = random.randint(0, nely)
            kl[i] = random.randint(0, nelz)

                
    if stype==1: #simple supported beam

        # use the area to do the random sampling as well as the use of the weighting_factor       
        for i in range(fnno):    
            il[i] = random.randint(0, nelx)
            jl[i] = nely
            kl[i] = random.randint(0, nelz)
  
    if stype==2: # constrained cantilever beam

        # use the area to do the random sampling as well as the use of the weighting_factor       
        for i in range(fnno):        

            il[i] = random.randint(0, nelx)
            jl[i] = nely
            kl[i] = random.randint(0, nelz)
 
                
    if stype==3: #x: [0,nelx] y:[nely/2,nely] z:[o,nelz]

        # use the area to do the random sampling as well as the use of the weighting_factor       
        for i in range(fnno):   
            il[i] = random.randint(0, nelx)
            jl[i] = nely
            kl[i] = random.randint(0, nelz)

            

    for i in range(fnno):  # 0 1 2 fnno-1
        fx[i] = random.randint(-10, 10)
        fy[i] = random.randint(-10, 10)
        fz[i] = random.randint(-10, 10)
        
        if fx[i] == 0 and fy[i] == 0 and fz[i] == 0:
            fy[i]=10

        loadnid[i] = kl[i] * (nelx + 1) * (nely + 1) + il[i] * (nely + 1) + (nely + 1 - jl[i])  # Node IDs


    print('x size:', nelx)
    print('y size:', nely)
    print('z size:', nelz)
    print('force position in x:', il)
    print('force position in y:', jl)
    print('force position in z:', kl)
    print('force in x direction:', fx)
    print('force in y direction:', fy)
    print('force in z direction:', fz)
    print('Volume fraction:',volfrac)
    
    return nelx,nely,nelz,il, jl, kl, fx, fy, fz, loadnid,volfrac,rmin