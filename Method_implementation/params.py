# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:55:25 2021

@author: e.christofi
"""

# Number of monomers per chain
nmer = 30

# Number of chains per configuration
nchain = 96

# Number of particles per configuration
npart = 11520

# Number of particles per chain
chainlen = 120

# Number of particles per monomer type
merlen_cis = 4
merlen_trans = 4
merlen_vinyl = 4

# Masses of particles per monomer type
masses_cis = [14.0270, 13.0190, 13.0190, 14.0270]
masses_trans = [14.0270, 13.0190, 13.0190, 14.0270]
masses_vinyl = [14.0270, 13.0190, 13.0190, 14.0270]

# Number of input spots per monomer type
dtseg_cis = 4
dtseg_trans = 4
dtseg_vinyl = 4

# Width of the input of the neural network
INPUT_WIDTH = 128


# Chains that end in vinyl-1,2 and contain cv3
cv3 = [0,3,5,7,8,9,11,12,13,17,18,19,20,23,31,35,36,38,39,43,45,46,53,59,62,65,67,68,69,70,74,75,76,77,79,83,84,86,87,88,91,95]

# List of indexes of the test-set configurations
com_list = [525, 1053, 1668, 1464, 396, 1688, 875, 1314, 1520, 458, 1, 985, 1441, 1575, 925, 1087, 1783, 1456, 1146, 108, 1095, 557, 340, 1494, 560, 116, 412, 1170, 1144, 1067, 780, 94, 760, 1054, 1304, 770, 1438, 1353, 49, 530, 1483, 901, 100, 483, 1429, 535, 264, 786, 742, 1787, 1843, 884, 1148, 17, 755, 612, 1769, 1164, 1617, 474, 906, 489, 140, 1329, 732, 635, 1432, 714, 1642, 266, 915, 250, 296, 383, 650, 355, 125, 68, 562, 1580, 800, 1065, 596, 83, 241, 407, 244, 72, 1215, 924, 1013, 1414, 893, 1651, 1595, 1003, 785, 1849, 1472, 235, 273, 444, 123, 1750, 1611, 486, 1641, 665, 1796, 122, 774, 43, 890, 993, 1847, 707, 1320, 1066, 844, 1760, 1623, 829, 1115, 854, 367, 1730, 1347, 1147, 1391, 1315, 898, 1829, 594, 1345, 401, 1568, 1279, 1381, 628, 1194, 306, 345, 1585, 1488, 228, 12, 1140, 699, 622, 720, 63, 1213, 1016, 224, 150, 1208, 160, 436, 1586, 1235, 1399, 353, 1258, 932, 364, 730, 1360, 615, 1339, 1005, 536, 942, 272, 1165, 1532, 1704, 1201, 1436, 1599, 236, 155, 1588, 501, 1652, 998]  


# ML hyper-parameters
EPOCHS = 1000
BATCH_SIZE = 32
