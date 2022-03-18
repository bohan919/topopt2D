# This script generates the 'strain energy field (at first iteration) -- support-free, TO-ed result' pairs


import top88_BohanAM
import top88_Bohan

# input: (nelx,nely,volfrac,penal,rmin,ft, path)
#       ft - 1: sensitivity filter
#            2: density filter
#            3: Heaviside Filter
#      'path' refers to the path with outputs are stored

